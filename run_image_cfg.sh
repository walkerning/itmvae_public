#!/bin/bash

set -e

if [[ $# -eq 0 ]]; then
    echo "Usage: VAE_RUNS=[number of runs(default: 1)] VAE_ADDITIONAL_ARG=[additional cmdine args(default:)] VAE_DATASET=[DATASET(default: mnist)] VAE_SAVE_MODEL=1/0 VAE_RANDOM_SEED=[SEED(default: None)] bash run_cfg.sh CFG_DIR [NAME(default: basename_of_CFG_DIR)] [RESULT_DIR(default: \"./results\")] [TOPIC_DIM(default: 50)]"
    exit
fi

export TF_CPP_MIN_LOG_LEVEL=3
save=${VAE_SAVE_MODEL:-0}
# SEED=${VAE_RANDOM_SEED:-"-1"}
# SEED=${VAE_RANDOM_SEED:-12345}
SEED=${VAE_RANDOM_SEED:-"-1"}
dataset=${VAE_DATASET:-mnist}
cfg_dir=${1}
d_name=$(basename ${cfg_dir})
name=${2:-${d_name}}
result_dir=${3:-"results"}
topic_dim=${4:-50}
additional_arg=${VAE_ADDITIONAL_ARG:-""}
runs=${VAE_RUNS:-1}
echo "dataset: ${dataset}; topic dim: ${topic_dim}"

this_result_dir=${result_dir}/${dataset}/${d_name}
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
perp_calc="0"
mean_calc="0"
median_calc="0"
mean_calc_5="0"
median_calc_5="0"
if [[ ${runs} -gt 1 ]]; then
    for i in `seq 1 ${runs}`;
    do
        log_file="${this_result_dir}/${name}_${i}.log"
        result_args=""
        if [[ ${save} -ne 0 ]]; then
            result_args="--save ${this_result_dir}/${name}_${i}/model --summary-dir ${this_result_dir}/${name}_${i}/summary"
        fi
        echo "$(date): RUN ${i}:"
        vae_image_run --topic-dim ${topic_dim} --cfg-file ${backup_cfg_dir}/model.yaml ${train_cfg_args} --dataset ${dataset} ${result_args} ${seed_args} ${additional_arg} >${log_file}

        perp=$(awk '/Test loss/ {print $NF}' ${log_file})
        echo "Test loss:  ${perp}"
        perp_calc=${perp_calc}+${perp}
    done
    echo ${perp_calc}
    echo ${mean_calc}
    echo ${median_calc}
    echo -n "Loss ${runs} RUNS: " && echo "scale=5; (${perp_calc})/${runs}" | bc
else
    vae_image_run --topic-dim ${topic_dim} --cfg-file ${backup_cfg_dir}/model.yaml ${train_cfg_args} --dataset ${dataset} ${result_args} ${seed_args} ${additional_arg} | tee ${log_file}
fi
