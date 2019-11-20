#!/bin/bash

set -e

if [[ $# -eq 0 ]]; then
    echo "Usage: TOPIC_DIM=[(default: 200)] VAE_LOAD=[pretrained model path(default:)] VAE_RUNS=[number of runs(default: 1)] VAE_ADDITIONAL_ARG=[additional cmdine args(default:)] VAE_DATASET=[DATASET(default: 20news)] VAE_SAVE_MODEL=1/0 VAE_RANDOM_SEED=[SEED(default: None)] bash run_cfg.sh CFG_DIR [NAME(default: basename_of_CFG_DIR)] [RESULT_DIR(default: \"./results\")]"
    exit
fi

export TF_CPP_MIN_LOG_LEVEL=3
save=${VAE_SAVE_MODEL:-1}
load=${VAE_LOAD:-""}
# SEED=${VAE_RANDOM_SEED:-"-1"}
# SEED=${VAE_RANDOM_SEED:-12345}
topic_dim=${TOPIC_DIM:-200}
SEED=${VAE_RANDOM_SEED:-"12345"}
dataset=${VAE_DATASET:-20news}
cfg_dir=${1}
d_name=$(basename ${cfg_dir})
name=${2:-${d_name}}
result_dir=${3:-"results"}
additional_arg=${VAE_ADDITIONAL_ARG:-""}
runs=${VAE_RUNS:-1}
echo "dataset: ${dataset}; topic dim: ${topic_dim}"

this_result_dir=${result_dir}/${dataset}/${d_name}
topic_file="${this_result_dir}/${name}.txt"
echo "topic file name: ${topic_file}"
log_file="${this_result_dir}/${name}.log"
echo "save result to: ${this_result_dir}; save log to: ${log_file}"
if [[ ! -d "${this_result_dir}" ]]; then
    mkdir -p ${this_result_dir}
fi

mkdir -p ${this_result_dir}/${name}
backup_cfg_dir=${this_result_dir}/${name}/config
if [[ -d "${backup_cfg_dir}" ]]; then
    rm -r "${backup_cfg_dir}"
fi
cp -r ${cfg_dir} ${backup_cfg_dir}

result_args=""
if [[ ${save} -ne 0 ]]; then
    result_args="--save ${this_result_dir}/${name}/model --summary-dir ${this_result_dir}/${name}/summary"
    mkdir -p ${this_result_dir}/${name}/tensors
    mkdir -p ${this_result_dir}/${name}/snapshots
fi

load_args=""
if [[ ! -z ${load} ]]; then
    load_args="--load ${load}"
fi

seed_args=""
if [[ ${SEED} -gt 0 ]]; then
    seed_args="--seed ${SEED}"
fi
train_cfg_file=${backup_cfg_dir}/train.yaml
train_cfg_args=""
if [[ -f "${train_cfg_file}" ]]; then
    train_cfg_args="--train-cfg-file ${train_cfg_file}"
fi
reader_args=""
context_size=20
if [[ ${dataset} == "rcv1_v2_gsm" ]]; then
    echo "Will use sparse reader for rcv1_v2_gsm dataset."
    reader_args="--reader-type index_sparse"
    context_size=0
fi
perp_calc="0"
mean_calc="0"
median_calc="0"
mean_calc_5="0"
median_calc_5="0"
if [[ ${runs} -gt 1 ]]; then
    for i in `seq 1 ${runs}`;
    do
        topic_file="${this_result_dir}/${name}_${i}.txt"
        log_file="${this_result_dir}/${name}_${i}.log"
        result_args=""
        if [[ ${save} -ne 0 ]]; then
            result_args="--save ${this_result_dir}/${name}_${i}/model --summary-dir ${this_result_dir}/${name}_${i}/summary"
	    mkdir -p ${this_result_dir}/${name}_${i}/tensors
	    mkdir -p ${this_result_dir}/${name}_${i}/snapshots
        fi
	ln -s $(readlink -f ${backup_cfg_dir}) ${this_result_dir}/${name}_${i}/config
        echo "$(date): RUN ${i}:"
        vae_topic_run --topic-dim ${topic_dim} --cfg-file ${backup_cfg_dir}/model.yaml ${train_cfg_args} --print-topic-file ${topic_file} --dataset ${dataset} ${result_args} ${seed_args} ${reader_args} ${additional_arg} ${load_args} >${log_file}
        bash ./run_npmi.sh ${topic_file} ${dataset} ${context_size} >/dev/null 2>&1

        perp=$(awk '/Test perplexity/ {print $NF}' ${log_file})
        echo "Test perplexity:  ${perp}"
        perp_calc=${perp_calc}+${perp}

        echo "TOP10:" && tail -n 2 "${topic_file}.oc"
        mean=$(tail -n 2 "${topic_file}.oc" | head -n 1 | awk '{print $NF}')
        median=$(tail -n 1 "${topic_file}.oc" | awk '{print $NF}')
        mean_calc=${mean_calc}+${mean}
        median_calc=${median_calc}+${median}

        echo "TOP5:" && tail -n 2 "${topic_file}.top5.oc"
        mean=$(tail -n 2 "${topic_file}.top5.oc" | head -n 1 | awk '{print $NF}')
        median=$(tail -n 1 "${topic_file}.top5.oc" | awk '{print $NF}')
        mean_calc_5=${mean_calc_5}+${mean}
        median_calc_5=${median_calc_5}+${median}
    done
    echo ${perp_calc}
    echo ${mean_calc}
    echo ${median_calc}
    echo -n "Perplexity ${runs} RUNS: " && echo "scale=5; (${perp_calc})/${runs}" | bc
    echo -n "TOP10 mean coherence across ${runs} RUNS: " && echo "scale=5; (${mean_calc})/${runs}" | bc
    echo -n "TOP10 median coherence across ${runs} RUNS: " && echo "scale=5; (${median_calc})/${runs}" | bc
    echo -n "TOP5 mean coherence across ${runs} RUNS: " && echo "scale=5; (${mean_calc_5})/${runs}" | bc
    echo -n "TOP5 median coherence across ${runs} RUNS: " && echo "scale=5; (${median_calc_5})/${runs}" | bc
else
    vae_topic_run --topic-dim ${topic_dim} --cfg-file ${backup_cfg_dir}/model.yaml ${train_cfg_args} --print-topic-file ${topic_file} --dataset ${dataset} ${result_args} ${seed_args} ${reader_args} ${additional_arg} ${load_args} | tee ${log_file}
    bash ./run_npmi.sh ${topic_file} ${dataset} ${context_size} >/dev/null 2>&1
    echo "TOP10:" && tail -n 2 "${topic_file}.oc"
    echo "TOP5:" && tail -n 2 "${topic_file}.top5.oc"
fi
