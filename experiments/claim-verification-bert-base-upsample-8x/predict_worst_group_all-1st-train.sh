#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=predict_worst-group
#SBATCH --out='predict_worst_group-dev-all-1st-train.log'
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex

pretrained='bert-base-uncased'
max_len=128
model_dir="${pretrained}-${max_len}-mod-1st-train"
inp_dir="${pretrained}-${max_len}-inp"
out_dir="${pretrained}-${max_len}-out"
group_out_dir="${out_dir}/groups-1st-train"
data_dir='../data'

unset -v latest

for file in "${model_dir}/checkpoints"/*.ckpt; do
  [[ $file -nt $latest ]] && latest=$file
done

if [[ -z "${latest}" ]]; then
  echo "Cannot find any checkpoint in ${model_dir}"
  exit
fi

echo "Latest checkpoint is ${latest}"

mkdir -p "${out_dir}"
mkdir -p "${group_out_dir}"

declare -a group_names=("negation_refutes" 
			"negation_supports" 
			"negation_nei" 
			"no_negation_refutes"  
			"no_negation_supports" 
			"no_negation_nei" 
			)
split='shared_task_dev'

for group_name in "${group_names[@]}"
do
  if [[ -f "${group_out_dir}/${group_name}.jsonl" ]]; then
    echo "Result '${group_out_dir}/${group_name}.jsonl' exists!"
    exit
  fi

  echo "Extracting group: '${group_name}'..."

  # extract worst groups
  # input: predicted output from postprocess_claim_verification.py 
  # "${out_dir}/${first_train}.jsonl" (not upsampled)
  python '../../extract_worst_groups.py' \
    --gold_file "${data_dir}/${split}.jsonl" \
    --pred_file "${out_dir}/${split}.jsonl" \
    --out_pred_file "${inp_dir}/${group_name}.tsv" \
    --out_gold_file "${group_out_dir}/${group_name}-gold.jsonl" \
    --out_pred_sent_file "${group_out_dir}/${group_name}-pred_sent.jsonl" \
    --corpus "${data_dir}/corpus.jsonl" \
    --worst_group_type "${group_name}"

  # predict R, S, or N, from training data and output probabilities
  python '../../predict.py' \
    --checkpoint_file "${latest}" \
    --in_file "${inp_dir}/${group_name}.tsv" \
    --out_file "${group_out_dir}/${group_name}.out" \
    --batch_size 128 \
    --gpus 1

  # get prediction sentences for predicted
  # make predicted file the same order as gold file for eval
  python '../../postprocess_claim_verification.py' \
    --data_file "${group_out_dir}/${group_name}-gold.jsonl" \
    --pred_sent_file "${group_out_dir}/${group_name}-pred_sent.jsonl" \
    --pred_claim_file "${group_out_dir}/${group_name}.out" \
    --out_file "${group_out_dir}/${group_name}.jsonl"

  python '../../eval_fever.py' \
    --gold_file "${group_out_dir}/${group_name}-gold.jsonl" \
    --pred_file "${group_out_dir}/${group_name}.jsonl" \
    --out_file "${group_out_dir}/eval.${group_name}.txt"
done
