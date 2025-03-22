#!/bin/bash

TRAINER=SAP
SHOTS=16
CFG=vit_b16_c2_ep50_batch4_4+4ctx

DATASET="ucf101"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

DATASET="dtd"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

DATASET="oxford_pets"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

DATASET="eurosat"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

DATASET="fgvc_aircraft"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

DATASET="caltech101"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

DATASET="oxford_flowers"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

DATASET="stanford_cars"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}

DATASET="food101"
bash scripts/parse_results.sh output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
