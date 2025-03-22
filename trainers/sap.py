import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import json
import re
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import pdb
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import DEFAULT_IMAGENET_TEMPLATE, IMAGENET_TEMPLATES #only 'a photo of a'
from utils.plotter import LossPlotter

_tokenizer = _Tokenizer()

def load_clip_to_cpu(cfg, zero_shot=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    #trainer can be ivlp, because encoder arch doesnt change
    if not zero_shot:
        design_details = {"trainer": 'IVLP',
                        "vision_depth": cfg.TRAINER.SAP.PROMPT_DEPTH_VISION,
                        "language_depth": cfg.TRAINER.SAP.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.SAP.N_CTX_VISION,
                        "language_ctx": cfg.TRAINER.SAP.N_CTX_TEXT}
    else:
        design_details = {"trainer": 'IVLP',
                        "vision_depth": 0, 
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0}
    model = clip.build_model(cfg,state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

def process_attributes(attribs, dset_name, cname, flag=False):
    if not flag:
        return attribs
    template_phrase = {'caltech-101':f'a photo of a {cname}', 'dtd':f'a photo of a {cname}',\
     'eurosat':f'a photo of a {cname}', 'fgvc_aircraft':f'a photo of a {cname}', 'food-101':f'a photo of a {cname}',\
     'oxford_flowers':f'a photo of a {cname}', 'oxford_pets':f'a photo of a {cname}','stanford_cars':f'a photo of a {cname}',\
     'sun397':f'a photo of a {cname}', 'ucf101':f'a photo of a {cname}', 'imagenet':f'a photo of a {cname}', 'imagenet-adversarial':f'a photo of a {cname}',\
     'imagenet-rendition':f'a photo of a {cname}','imagenet-sketch':f'a photo of a {cname}'}
    out_attribs = []
    for attrib in attribs:
        try:
            end_phrase = attrib.split(',',1)[1].strip()
        except:
            words = attrib.split(' ')
            end_phrase = ('which ' if words[0] in ['is', 'has'] else 'which has ') + re.sub('[,\.]*','', ' '.join(words).strip())
        out_attribs.append(template_phrase[dset_name] + ", " + end_phrase+".")
    return out_attribs 

class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, attribute_fpath, clip_model):
        super().__init__()
        # Make sure Language depth >= 1
        assert cfg.TRAINER.SAP.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch  "
        n_ctx = cfg.TRAINER.SAP.N_CTX_TEXT
        ctx_init = cfg.TRAINER.SAP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print(f"Independent V-L design")
        print(f'Initial text context: "{prompt_prefix}"')
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.SAP.N_CTX_VISION}")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        with open(osp.join(attribute_fpath, 'short_attributes.json')) as attrfile:
            attrs = json.load(attrfile)
            all_attrs_list = []; all_attrs_list_template = []
            class_attr_counts = []
            cumucount = 0
            dset_name = osp.basename(osp.normpath(attribute_fpath))
            for ii, cname in enumerate(classnames):
                if dset_name == 'imagenet':
                    attribs = attrs[cname.replace("_"," ").lower() if dset_name!='fgvc_aircraft' else cname.replace("_"," ")][0:3]
                else:
                    attribs = attrs[cname.replace("_"," ").lower() if dset_name!='fgvc_aircraft' else cname.replace("_"," ")]
                template_attribs = process_attributes(attribs, dset_name, cname.replace("_"," ").lower(), True) #use this for prompt
                all_attrs_list.extend(attribs); all_attrs_list_template.extend(template_attribs)
                class_attr_counts.append(torch.arange(cumucount, cumucount+len(attribs)))
                cumucount += len(attribs)

        self.all_attrs_list = all_attrs_list
        self.all_attrs_list_template = all_attrs_list_template
        self.class_attr_mask = torch.zeros(len(classnames), len(all_attrs_list)).type(dtype).cuda()
        for ii in range(len(classnames)):
            self.class_attr_mask[ii, class_attr_counts[ii]] = 1

        prompts = self.all_attrs_list_template
        n_cls = len(prompts)
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]
        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
        return prompts

def stdnorm(x):
    #perform stadard normalization with bessel's correction
    mu = x.mean(dim=-1, keepdim=True)
    sigma = x.std(dim=-1, keepdim=True)
    if sigma.min().item()<1e-06:
        mask = sigma < 1e-06
        sigma[mask] = 1.

    return (x - mu)/sigma

def entropy(x, e=1e-09):
    #fp16 entropy computation along last dimension
    return (-x.float() * torch.log(x.float() + e)).sum(dim=-1)


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, attribute_fpath, clip_model):
        super().__init__()
        self.cfg = cfg
        self.prompt_learner = VLPromptLearner(cfg, classnames, attribute_fpath, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        base_clip_model_image = load_clip_to_cpu(cfg, True)
        self.base_image_encoder = base_clip_model_image.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        #image post projection
        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj
        reduction = cfg.TRAINER.SAP.REDUCTION
        ipdim = self.proj.shape[0]; opdim = self.proj.shape[1] 
        mybias = torch.zeros(opdim).type(self.dtype)
        self.mybias = nn.Parameter(mybias)
        #adding hook to image encoder 
        self.handles = []
        self.intermediate_layers = [12]
        self.intermediate_image_features = {}
        def set_hooks(mname):
            def hook(module, input, output):
                self.intermediate_image_features[mname] = output
            return hook
        for mname, mod in self.image_encoder.named_modules():
            if mname in [f'transformer.resblocks.{ii-1}' for ii in self.intermediate_layers]:
                self.handles.append(mod.register_forward_hook(set_hooks(mname)))

        #use base clip model to get attr features
        with torch.no_grad():
            #storing attribute features
            all_attrs_list = self.prompt_learner.all_attrs_list
            tokenized_attrs = torch.cat([clip.tokenize(a) for a in all_attrs_list], dim=0)
            base_clip_model = load_clip_to_cpu(cfg, True)
            base_clip_model.float().cuda() #TODO: add back .float() 
            self.attr_features = base_clip_model.encode_text(tokenized_attrs.cuda())
            #storing text features
            p = self.prompt_learner.all_attrs_list_template
            tok_p = torch.cat([clip.tokenize(c) for c in p], dim=0)
            self.base_text_features = base_clip_model.encode_text(tok_p.cuda())
            base_clip_model.cpu()

    
    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)
        _ = self.image_encoder(image.type(self.dtype))

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #cross attention between the intermediate features and attribute features
        cross_image_features_list = []
        global_image_features_list = []
        for layer in self.intermediate_layers:
            intermediate_image_features_ = self.intermediate_image_features[f'transformer.resblocks.{layer-1}']
            #transpose, project and normalize
            intermediate_image_features_ = torch.transpose(intermediate_image_features_, 0, 1)
            #intermediate_image_features = self.myproj(intermediate_image_features[:, 1:1 + 196, :])
            intermediate_image_features_ = self.ln_post(intermediate_image_features_[:, :, :])
            Bshape = intermediate_image_features_.shape[0]; Fshape = intermediate_image_features_.shape[1]; Dshape = self.proj.shape[1]
            intermediate_image_features_ = intermediate_image_features_ @ self.proj +\
                torch.cat([torch.zeros(Bshape, 1, Dshape).type(self.dtype).cuda(), self.mybias.view(1,1,-1).expand(Bshape,Fshape-1,Dshape)], dim=1)
            #intermediate_image_features_ = intermediate_image_features_ / intermediate_image_features_.norm(dim=-1, keepdim=True)
            global_image_features_ = intermediate_image_features_[:,0:1,:]
            local_image_features_ = intermediate_image_features_[:,1:1+196,:]
            #get attribute relevance scores
            attr_relevance_score = F.softmax(global_image_features_.squeeze(1) @ self.attr_features.type(self.dtype).t()) #B x P
            #attention map over local features, get local features
            image_cross_att_logits = torch.bmm(self.attr_features.unsqueeze(0).expand(local_image_features_.shape[0],-1,-1).type(self.dtype), local_image_features_.transpose(2,1))
            image_cross_att = F.softmax(image_cross_att_logits, dim=-1)
            cross_image_features_ = torch.bmm(image_cross_att, local_image_features_) #B x P x 512
            #get weights for global-local fusion
            local_score_ = (torch.max(image_cross_att, dim=-1, keepdim=False)[0]).type(self.dtype) #weights for local feature B x P
            local_score = (attr_relevance_score * local_score_).sum(dim=-1) #final local_score
            #first weight attribute specific local features 
            final_cross_image_features_ = torch.bmm(attr_relevance_score.unsqueeze(1), cross_image_features_) #B x 1 x 512
            final_cross_image_features_ = (1-local_score).view(-1,1,1) * global_image_features_ + local_score.view(-1,1,1) * final_cross_image_features_
            
            cross_image_features_list.append(final_cross_image_features_)
            global_image_features_list.append(global_image_features_)
        
        cross_image_features = torch.cat(cross_image_features_list, dim=1).mean(dim=1, keepdim=False)
        cross_image_features = cross_image_features / cross_image_features.norm(dim=-1, keepdim=True)

        global_image_features = torch.cat(global_image_features_list, dim=1).mean(dim=1, keepdim=False)
        global_image_features = global_image_features/ global_image_features.norm(dim=-1, keepdim=True)

        logits_ = logit_scale* (cross_image_features @ text_features.t())
        logits_ = logits_ @ (self.prompt_learner.class_attr_mask/self.prompt_learner.class_attr_mask.sum(dim=-1, keepdim=True)).t()
        
        if self.prompt_learner.training:
            with torch.no_grad():
                base_image_features = self.base_image_encoder(image.type(self.dtype))
                base_text_features = self.base_text_features

                base_image_features = base_image_features/base_image_features.norm(dim=-1, keepdim=True)
                base_text_features = base_text_features/base_text_features.norm(dim=-1, keepdim=True)
                
            loss_ce = F.cross_entropy(logits_, label)
            return base_image_features, global_image_features, base_text_features, text_features, loss_ce
        return logits_

    def remove_hooks(self):
        for h in self.handles:
            h.remove()

    def only_text_model(self, tokenized_prompts):
        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)



@TRAINER_REGISTRY.register()
class SAP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.SAP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        attribute_fpath = "give the path to the attributes file"

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.SAP.PREC == "fp32" or cfg.TRAINER.SAP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, attribute_fpath, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name or "mybias" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Number of trainable params: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.SAP.PREC == "amp" else None

        #plotter
        self.plotter = LossPlotter(cfg, len(self.train_loader_x), False)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.SAP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
        
            base_image_features, cross_image_features, base_text_features,\
                text_features, loss_ce =  model(image, label)
            l1_text_loss = F.l1_loss(base_text_features, text_features, reduction='mean') * self.cfg.TRAINER.SAP.TEXT_LOSS_WEIGHT
            l1_image_loss = F.l1_loss(base_image_features, cross_image_features, reduction='mean') * self.cfg.TRAINER.SAP.IMAGE_LOSS_WEIGHT
            kwargs = {'l1_image_loss': l1_image_loss.item(), 'l1_text_loss': l1_text_loss.item()}
            self.plotter.save(**kwargs)

            loss = loss_ce + (l1_text_loss + l1_image_loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
