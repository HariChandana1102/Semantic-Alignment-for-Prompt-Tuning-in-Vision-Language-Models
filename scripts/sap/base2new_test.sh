#!/bin/bash

#cd ../..

# custom config
DATA=path to your dataset
TRAINER=SAP

GPU=$1
DATASET=$2
SEED=$3
CFG=vit_b16_c2_ep20_batch4_4+4ctx
dg=False

SHOTS=16
LOADEP=20
SUB=new
GRADCAM=False

if [ ${dg} == "true" ]; then
    COMMON_DIR=imagenet/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
else
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
fi
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}

echo "Evaluating model"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
--model-dir ${MODEL_DIR} \
--load-epoch ${LOADEP} \
--eval-only \
TRAIN.GRADCAM ${GRADCAM} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
