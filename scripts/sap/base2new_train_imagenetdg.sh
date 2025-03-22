#!/bin/bash

#cd ../..

# custom config
DATA=path to your dataset
TRAINER=SAP

GPU=$1
DATASET=imagenet
SEED=1
CFG=$2

SHOTS=16
SUB=all

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
