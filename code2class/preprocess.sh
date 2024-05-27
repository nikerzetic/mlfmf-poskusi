#!/usr/bin/env bash

###########################################################
RAW_DATA_DIR=data    # where the un-preprocessed data files are
DATA_DIR=data/code2class      # where the preprocessed data files are written to
                              # (value must be different thant RAW_DATA_DIR!)
DATASET_NAME=stdlib           # name of dataset
MAX_CONTEXTS=200              # max data contexts to use in training
TOKEN_VOCAB_SIZE=186277       # the number of tokens and target words to keep ..
TARGET_VOCAB_SIZE=26347       # .. in the vocabulary (the top occurring words and paths will be kept).
PYTHON=python
###########################################################

TRAIN_DATA_FILE=${RAW_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.train.raw.txt
VAL_DATA_FILE=${RAW_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.val.raw.txt
TEST_DATA_FILE=${RAW_DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.test.raw.txt

mkdir -p ${DATA_DIR}
mkdir -p ${DATA_DIR}/${DATASET_NAME}

TEMP_JOINED_FILE=${DATA_DIR}/${DATASET_NAME}/temp.joined.raw.txt
echo "Concatenating to one file"
cat ${TRAIN_DATA_FILE} ${VAL_DATA_FILE} ${TEST_DATA_FILE} > ${TEMP_JOINED_FILE}

TARGET_HISTOGRAM_FILE=${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.tgt.c2c
SOURCE_TOKEN_HISTOGRAM=${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.ori.c2c
NODE_HISTOGRAM_FILE=${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}.histo.node.c2c

echo "Creating histograms from the joined data"
cat ${TEMP_JOINED_FILE} | cut -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TEMP_JOINED_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${SOURCE_TOKEN_HISTOGRAM}
cat ${TEMP_JOINED_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${NODE_HISTOGRAM_FILE}

rm -rf ${TEMP_JOINED_FILE}

echo "Running Preprocess.py (Creating Dictionary and format files)"
${PYTHON} code2class/preprocess.py --train_data ${TRAIN_DATA_FILE} --test_data ${TEST_DATA_FILE} --val_data ${VAL_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --token_vocab_size ${TOKEN_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --token_histogram ${SOURCE_TOKEN_HISTOGRAM} \
  --node_histogram ${NODE_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} \
  --output_name ${DATA_DIR}/${DATASET_NAME}/${DATASET_NAME}
