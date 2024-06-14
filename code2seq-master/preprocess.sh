#!/usr/bin/env bash
###########################################################
# Change the following values to preprocess a new dataset.
# TRAIN_DIR, VAL_DIR and TEST_DIR should be paths to      
#   directories containing sub-directories with .java files
# DATASET_NAME is just a name for the currently extracted 
#   dataset.                                              
# MAX_DATA_CONTEXTS is the number of contexts to keep in the dataset for each 
#   method (by default 1000). At training time, these contexts
#   will be downsampled dynamically to MAX_CONTEXTS.
# MAX_CONTEXTS - the number of actual contexts (by default 200) 
# that are taken into consideration (out of MAX_DATA_CONTEXTS)
# every training iteration. To avoid randomness at test time, 
# for the test and validation sets only MAX_CONTEXTS contexts are kept 
# (while for training, MAX_DATA_CONTEXTS are kept and MAX_CONTEXTS are
# selected dynamically during training).
# SUBTOKEN_VOCAB_SIZE, TARGET_VOCAB_SIZE -   
#   - the number of subtokens and target words to keep 
#   in the vocabulary (the top occurring words and paths will be kept). 
# NUM_THREADS - the number of parallel threads to use. It is 
#   recommended to use a multi-core machine for the preprocessing 
#   step and set this value to the number of cores.
# PYTHON - python3 interpreter alias.
TRAIN_DIR=data/raw/stdlib/code2vec/train
VAL_DIR=data/raw/stdlib/code2vec/val
TEST_DIR=data/raw/stdlib/code2vec/test
DUMP_DIR=data/raw/stdlib/code2vec
DATASET_NAME=stdlib
MAX_DATA_CONTEXTS=1000
MAX_CONTEXTS=200
SUBTOKEN_VOCAB_SIZE=186277
TARGET_VOCAB_SIZE=26347
NUM_THREADS=32
PYTHON=python3.11
###########################################################

DIR=data/code2seq/${DATASET_NAME}

mkdir -p data
mkdir -p ${DIR}

TRAIN_DATA_FILE=${DIR}/${DATASET_NAME}.train.raw.txt
VAL_DATA_FILE=${DIR}/${DATASET_NAME}.val.raw.txt
TEST_DATA_FILE=${DIR}/${DATASET_NAME}.test.raw.txt

echo "Extracting paths from validation set..."
${PYTHON} my_extract.py --dir ${VAL_DIR} --max_path_length 16 --max_path_width 3 --num_threads ${NUM_THREADS} --out_file ${VAL_DATA_FILE}
echo "Finished extracting paths from validation set"
echo "Extracting paths from test set..."
${PYTHON} my_extract.py --dir ${TEST_DIR} --max_path_length 16 --max_path_width 3 --num_threads ${NUM_THREADS} --out_file ${TEST_DATA_FILE}
echo "Finished extracting paths from test set"
echo "Extracting paths from training set..."
${PYTHON} my_extract.py --dir ${TRAIN_DIR} --max_path_length 16 --max_path_width 3 --num_threads ${NUM_THREADS} --out_file ${TRAIN_DATA_FILE}
echo "Finished extracting paths from training set"

TARGET_HISTOGRAM_FILE=${DIR}/${DATASET_NAME}.histo.tgt.c2s
SOURCE_SUBTOKEN_HISTOGRAM=${DIR}/${DATASET_NAME}.histo.ori.c2s
NODE_HISTOGRAM_FILE=${DIR}/${DATASET_NAME}.histo.node.c2s

echo "Creating histograms from the training data"
cat ${TRAIN_DATA_FILE} | cut -d' ' -f1 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${TARGET_HISTOGRAM_FILE}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f1,3 | tr ',|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${SOURCE_SUBTOKEN_HISTOGRAM}
cat ${TRAIN_DATA_FILE} | cut -d' ' -f2- | tr ' ' '\n' | cut -d',' -f2 | tr '|' '\n' | awk '{n[$0]++} END {for (i in n) print i,n[i]}' > ${NODE_HISTOGRAM_FILE}

${PYTHON} code2seq-master/preprocess.py --train_data ${TRAIN_DATA_FILE} --test_data ${TEST_DATA_FILE} --val_data ${VAL_DATA_FILE} \
  --max_contexts ${MAX_CONTEXTS} --max_data_contexts ${MAX_DATA_CONTEXTS} --subtoken_vocab_size ${SUBTOKEN_VOCAB_SIZE} \
  --target_vocab_size ${TARGET_VOCAB_SIZE} --subtoken_histogram ${SOURCE_SUBTOKEN_HISTOGRAM} \
  --node_histogram ${NODE_HISTOGRAM_FILE} --target_histogram ${TARGET_HISTOGRAM_FILE} --output_name ${DIR}/${DATASET_NAME}

ALT_TRAIN=${DIR}/train.c2s
ALT_TEST=${DIR}/test.c2s
ALT_VAL=${DIR}/val.c2s
ALT_PREDICT=${DIR}/predict.c2s

cp ${DIR}/${DATASET_NAME}.train.c2s ${ALT_TRAIN}
cp ${DIR}/${DATASET_NAME}.test.c2s ${ALT_TEST}
cp ${DIR}/${DATASET_NAME}.val.c2s ${ALT_VAL}
cat ${ALT_TRAIN} ${ALT_TEST} ${ALT_VAL} > ${ALT_PREDICT}
    
# If all went well, the raw data files can be deleted, because preprocess.py creates new files 
# with truncated and padded number of paths for each example.
rm ${TRAIN_DATA_FILE} ${VAL_DATA_FILE} ${TEST_DATA_FILE} ${TARGET_HISTOGRAM_FILE} ${SOURCE_SUBTOKEN_HISTOGRAM} \
  ${NODE_HISTOGRAM_FILE}

