#!/bin/bash

# script that computes the observed coherence (pointwise mutual information, normalised pmi or log 
# conditional probability)
# steps:
# 1. sample the word counts of the topic words based on the reference corpus
# 2. compute the observed coherence using the chosen metric

pid=$$
sub_pid=0
function kill_subprocesses() {
    if [[ ${sub_pid} -ne 0 ]]; then
    	pkill -P ${sub_pid}
    fi
    pkill -P ${pid}
    exit
}

trap 'kill_subprocesses' 2

# parameters
metric="npmi" #evaluation metric: pmi, npmi or lcp

set -e

if [[ $# -eq 0 ]]; then
    echo "Usage: bash run_npmi.sh TOPIC_FILE [DATASET(default: 20news)] [CONTEXT_SIZE(default: 20)] [OUTPUT(default: TOPIC_FILE.oc)]"
    exit
fi

topic_file=${1}
name=$(basename ${1})
dataset=${2:-"20news"}
# context_size=${3:-0}
context_size=${3:-20}

here=$( readlink -f $( dirname "${BASH_SOURCE[0]}" ) )
ref_corpus_dir="${here}/vae_topicmodel/datasets/${dataset}/ref_corpus"
oc_file=${4:-${topic_file}.oc}
oc_file_top5=${5:-${topic_file}.top5.oc}

echo "Use reference corpus in directory ${ref_corpus_dir}"
echo "Result will be write to ${oc_file}"

# output
wordcount_file=$(tempfile)

#compute the word occurrences
echo "TOP10: Computing word occurrence..."
python ${here}/topic_interpretability/ComputeWordCount.py $topic_file $ref_corpus_dir --context-size ${context_size} > $wordcount_file &
sub_pid=$!
wait

#compute the topic observed coherence
echo "TOP10: Computing the observed coherence..."
python ${here}/topic_interpretability/ComputeObservedCoherence.py $topic_file $metric $wordcount_file > $oc_file &
sub_pid=$!

# top-5
wordcount_file=$(tempfile)
#compute the word occurrences
echo "TOP5: Computing word occurrence..."
python ${here}/topic_interpretability/ComputeWordCount.py <(cat ${topic_file} | cut -d' ' -f1,2,3,4,5) $ref_corpus_dir --context-size ${context_size} > $wordcount_file &
sub_pid=$!
wait

#compute the topic observed coherence
echo "TOP5: Computing the observed coherence..."
python ${here}/topic_interpretability/ComputeObservedCoherence.py <(cat ${topic_file} | cut -d' ' -f1,2,3,4,5) $metric $wordcount_file > $oc_file_top5 &
sub_pid=$!
wait
