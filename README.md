# 🌟 Semantic Alignment for Prompt Tuning in Vision-Language Models

📢 **Our work has been accepted at TMLR!**   
🔗 [**OpenReview Link**](https://openreview.net/forum?id=avDr56QjSI)  

## 📌 Authors  
[Hari Chandana Kuchibhotla](https://sites.google.com/view/hari-chandana-kuchibhotla/home), [Sai Srinivas Kancheti](https://ksais.github.io/), [Abbavaram Gowtham Reddy](https://gautam0707.github.io/), and [Vineeth N Balasubramanian](https://people.iith.ac.in/vineethnb/) <br>
📍 [Indian Institute of Technology Hyderabad (IITH)][https://www.iith.ac.in/]


## 🔍 Abstract  
Going beyond mere fine-tuning of vision-language models (VLMs), learnable **prompt tuning** has emerged as a promising, resource-efficient alternative. Despite their potential, effective prompt tuning faces the following challenges:

✅ Overfitting in low-shot scenarios, limiting adaptability and performance on novel classes.  
✅ Dependence on the label space, leading to weaker performance in large class spaces.  

We introduce **SAP (Semantic Alignment for Prompt-Tuning)**, a novel method that leverages **class descriptions from Large Language Models (LLMs)** to bridge image and text modalities. SAP constructs **part-level description-guided image and text features**, aligning them for more **generalizable prompts**.  

Our extensive experiments across **11 benchmark datasets** demonstrate that **SAP significantly outperforms existing methods**. 🚀  

---

## 🛠 Installation  
This codebase has been tested on **Ubuntu 20.04** with **Python 3.8**.

### 🔹 Setup Conda Environment  
```bash
# Create and activate conda environment
conda create -y -n sap python=3.8
conda activate sap

# Install PyTorch (requires version >= 1.8.1)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### 🔹 Installing Dassl Library
```bash
# Clone Dassl repository
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install Dassl
python setup.py develop
cd ..
```

### 🔹 Clone SAP repository and Install requirements
```bash
# Clone SAP repository
git clone https://github.com/HariChandana1102/Semantic-Alignment-for-Prompt-Tuning-in-Vision-Language-Models.git
cd Semantic-Alignment-for-Prompt-Tuning-in-Vision-Language-Models

# Install required packages
pip install -r requirements.txt

# Update setuptools
pip install setuptools==59.5.0
```
### 🔹 Datasets
For dataset setup and download, please refer to this [link](https://github.com/muzairkhattak/PromptSRC/blob/main/docs/DATASETS.md)

### 🚀 Training & Evaluation
🔹 Base-to-Novel Setting<br>
To train and test in the Base-to-Novel setting, use:
```bash
bash scripts/sap/base2new_train.sh ${GPU} ${dataset} ${seed}
bash scripts/sap/base2new_test.sh ${GPU} ${dataset} ${seed}
```
For example:
```bash
bash scripts/sap/base2new_train.sh 0 eurosat 1
```

🔹 Generalized Zero-Shot (GZS) Setting <br>
Modify SUB=new to SUB=all in the scripts to run in GZS mode.

📜 Citation<br>
If you find our work useful, please consider citing us:
```bash
@article{SAP,
  title={Semantic Alignment for Prompt Tuning in Vision-Language Models},
  author={Hari Chandana Kuchibhotla, Sai Srinivas Kancheti, Abbavaram Gowtham Reddy and Vineeth N Balasubramanian},
  journal={TMLR},
  year={2024}
}
```
### Acknowledgement: This code repository is built on [PromptSRC](https://github.com/muzairkhattak/PromptSRC/tree/main)
