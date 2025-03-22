#!/bin/bash

#this script makes a copy of the most recent log file and calls parse_test_results.py

DIR=$1

for FOLDER in $(ls -l ${DIR} | grep -iEo 'seed.*')
do
    LOG_FOLDER=${DIR}/${FOLDER}
    LOG_FILE=$(ls -t ${LOG_FOLDER} | grep -iE "log\.txt.*" | head -1 | awk '{print $1}')
    #if LOG_FILE is 'log.txt' proceed, else make a copy
    if [[ "$LOG_FILE" != 'log.txt' ]]; then
        cp -f "${LOG_FOLDER}/${LOG_FILE}" "${LOG_FOLDER}/log.txt"
        echo "Copy of ${LOG_FILE} made."
    fi
done

python parse_test_res.py ${DIR} --test-log