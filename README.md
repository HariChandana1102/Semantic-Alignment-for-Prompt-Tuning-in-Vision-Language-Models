# Semantic-Alignment-for-Prompt-Tuning-in-Vision-Language-Models

# Installation

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
# Datasets
* For downloading the datasets, please follow https://github.com/muzairkhattak/PromptSRC/blob/main/docs/DATASETS.md
  Once the datasets are downloaded, copy the semantic descriptions of the respective dataset into its folder.

# Training and Evaluation
* For training and testing in the Base-to-Novel setting, run the following command
  ```bash
  bash scripts/sap/base2new_train.sh ${GPU} ${dataset} ${seed}
  bash scripts/sap/base2new_test.sh ${GPU} ${dataset} ${seed}

  # For example:  bash scripts/sap/base2new_train.sh 0 eurosat 1
  ```
  To run the code in the GZS setting, change the SUB=new to SUB=all

##### Acknowledgement: This readme file for installation and package requirements follows the PromptSRC (https://github.com/muzairkhattak/PromptSRC/blob/main/docs/INSTALL.md) official repository.

