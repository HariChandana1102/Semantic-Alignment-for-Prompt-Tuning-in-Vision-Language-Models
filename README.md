# Semantic-Alignment-for-Prompt-Tuning-in-Vision-Language-Models
Work done by Hari Chandana Kuchibhotla, Sai Srinivas Kancheti, Abbavaram Gowtham Reddy, and Vineeth N Balasubramanian at Indian Institute of Technology Hyderabad (IITH)

Abstract: Going beyond mere fine-tuning of vision-language models (VLMs), learnable prompt tuning has emerged as a promising, resource-efficient alternative. Despite their potential, effectively learning prompts faces the following challenges: (i) training in a low-shot scenario results in overfitting, limiting adaptability, and yielding weaker performance on newer classes or datasets; (ii) prompt-tuning's efficacy heavily relies on the label space, with decreased performance in large class spaces, signaling potential gaps in bridging image and class concepts. In this work, we investigate whether better text semantics can help address these concerns. In particular, we introduce a prompt-tuning method that leverages class descriptions obtained from Large Language Models (LLMs). These class descriptions are used to bridge image and text modalities. Our approach constructs part-level description-guided image and text features, which are subsequently aligned to learn more generalizable prompts. Our comprehensive experiments conducted across 11 benchmark datasets show that our method outperforms established methods, demonstrating substantial improvements.





## Installation

This codebase is tested on Ubuntu 20.04.2 LTS with Python 3.8. Follow the below steps to create the environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n sap python=3.8

# Activate the environment
conda activate sap

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone SAP code repository and install requirements
```bash
# Clone SAP code base
git clone https://github.com/HariChandana1102/Semantic-Alignment-for-Prompt-Tuning-in-Vision-Language-Models.git

cd SAP/
# Install requirements

pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0
```
## Datasets
* For downloading the datasets, please follow https://github.com/muzairkhattak/PromptSRC/blob/main/docs/DATASETS.md.


## Training and Evaluation
* For training and testing in the Base-to-Novel setting, run the following command
  ```bash
  bash scripts/sap/base2new_train.sh ${GPU} ${dataset} ${seed}
  bash scripts/sap/base2new_test.sh ${GPU} ${dataset} ${seed}

  # For example:  bash scripts/sap/base2new_train.sh 0 eurosat 1
  ```
  To run the code in the GZS setting, change the SUB=new to SUB=all


## Citation
If you find our work interesting or use it to develop it further, please cite us
```
@article{SAP,
  title={Semantic Alignment for Prompt Tuning in Vision-Language Models},
  author={Hari Chandana Kuchibhotla, Sai Srinivas Kancheti, Abbavaram Gowtham Reddy and Vineeth N Balasubramanian},
  journal={TMLR},
  year={2024}
}
```
#### Acknowledgement: This readme file for installation and package requirements follows the PromptSRC (https://github.com/muzairkhattak/PromptSRC/blob/main/docs/INSTALL.md) official repository.

