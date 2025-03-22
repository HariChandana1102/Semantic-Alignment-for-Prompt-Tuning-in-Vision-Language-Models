#!/bin/bash

#cd ../..

# custom config
DATA=path to your dataset
TRAINER=SAP

GPU=$1
DATASET=$2
SEED=$3
CFG=vit_b16_c2_ep20_batch4_4+4ctx

SHOTS=16
SUB=base   ######change for few-shot hyperparam tuning ########


DIR=output/base2new/train_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}

echo "Runing the first phase job and save the output to ${DIR}"

CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.NUM_SHOTS ${SHOTS} \
DATASET.SUBSAMPLE_CLASSES ${SUB}
