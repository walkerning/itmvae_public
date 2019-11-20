#!/bin/bash

if [[ $# -lt 2 ]]; then
    echo "Usage: bash run_cfg.sh MODEL_DIR CFG_DIR [DATASET(default: 20news)] [CONTEXT_SIZE(default: 20)] [TOPIC_DIM(default: 50)]"
    exit
fi

export TF_CPP_MIN_LOG_LEVEL=3
load=${1}
cfg_dir=${2}
dataset=${3:-20news}
context_size=${4:-20}
topic_dim=${5:-50}
echo "Construct model from config dir ${cfg_dir}; Load model from ${load}."

vae_topic_runtest ${load}/model --topic-dim ${topic_dim} --cfg-file ${cfg_dir}/model.yaml --train-cfg-file ${cfg_dir}/train.yaml --dataset ${dataset} --print-topic-file ${load}/test.txt

bash ./run_npmi.sh ${load}/test.txt ${dataset} ${context_size} >/dev/null 2>&1

tail -n 2 ${load}/test.txt.oc

tail -n 2 ${load}/test.txt.top5.oc
