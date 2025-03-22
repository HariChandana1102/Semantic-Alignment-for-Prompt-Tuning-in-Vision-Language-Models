GPU=$1
DATASET=$2
SEEDS=("$@")

SHOTS=16
TRAINER=SAP
CFG=vit_b16_c2_ep20_batch4_4+4ctx
dg=False
SUB=base

for seed in ${SEEDS[@]:2}
do
    bash scripts/sap/base2new_train.sh ${GPU} ${DATASET} ${seed} ${CFG} 2>/dev/null
    bash scripts/sap/base2new_test.sh ${GPU} ${DATASET} ${seed} ${CFG} 2>/dev/null
done


bash scripts/parse_results.sh output/base2new/train_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
bash scripts/parse_results.sh output/base2new/test_new/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
