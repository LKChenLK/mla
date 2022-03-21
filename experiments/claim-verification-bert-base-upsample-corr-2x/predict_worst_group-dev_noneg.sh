#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=predict_worst-group
#SBATCH --out='predict_worst_group-dev-noneg.log'
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
model_dir="${pretrained}-${max_len}-mod"
inp_dir="${pretrained}-${max_len}-inp"
out_dir="${pretrained}-${max_len}-out"

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

split='shared_task_dev'
out_name='worst_group_dev-noneg'
worst_group_type='no_negation'

if [[ -f "${out_dir}/${out_name}.jsonl" ]]; then
  echo "Result '${out_dir}/${out_name}.jsonl' exists!"
  exit
fi

# extract worst groups
# input: predicted output from postprocess_claim_verification.py 
# "${out_dir}/${first_train}.jsonl" (not upsampled)
python '../../extract_worst_groups.py' \
  --gold_file "${data_dir}/${split}.jsonl" \
  --pred_file "${out_dir}/${split}.jsonl" \
  --out_pred_file "${inp_dir}/${out_name}.tsv" \
  --out_gold_file "${data_dir}/${out_name}-gold.jsonl" \
  --out_pred_sent_file "${data_dir}/${out_name}-pred_sent.jsonl" \
  --corpus "${data_dir}/corpus.jsonl" \
  --worst_group_type "${worst_group_type}"

# predict R, S, or N, from training data and output probabilities
python '../../predict.py' \
  --checkpoint_file "${latest}" \
  --in_file "${inp_dir}/${out_name}.tsv" \
  --out_file "${out_dir}/${out_name}.out" \
  --batch_size 128 \
  --gpus 1

# get prediction sentences for predicted
# make predicted file the same order as gold file for eval
python '../../postprocess_claim_verification.py' \
  --data_file "${data_dir}/${out_name}-gold.jsonl" \
  --pred_sent_file "${data_dir}/${out_name}-pred_sent.jsonl" \
  --pred_claim_file "${out_dir}/${out_name}.out" \
  --out_file "${out_dir}/${out_name}.jsonl"

python '../../eval_fever.py' \
  --gold_file "${data_dir}/${out_name}-gold.jsonl" \
  --pred_file "${out_dir}/${out_name}.jsonl" \
  --out_file "${out_dir}/eval.${out_name}.txt"

