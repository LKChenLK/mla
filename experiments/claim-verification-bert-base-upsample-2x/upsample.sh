#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=upsample_bert-base
#SBATCH --out='upsample_bert-base.log'
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
inp_dir="${pretrained}-${max_len}-inp"
model_dir="${pretrained}-${max_len}-mod-1st-training"
out_dir="${pretrained}-${max_len}-out"

data_dir='../data'
pred_sent_dir='../sentence-selection/bert-base-uncased-128-out'

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

split='first_train'

if [[ -f "${out_dir}/${split}.jsonl" ]]; then
  echo "Result '${out_dir}/${split}.jsonl' exists!"
  exit
fi

# get sentences for training examples
python '../../preprocess_claim_verification.py' \
  --corpus "${data_dir}/corpus.jsonl" \
  --in_file "${pred_sent_dir}/train.jsonl" \
  --out_file "${inp_dir}/${split}.tsv"

# predict R, S, or N, from training data and output probabilities
python '../../predict.py' \
  --checkpoint_file "${latest}" \
  --in_file "${inp_dir}/${split}.tsv" \
  --out_file "${out_dir}/${split}.out" \
  --batch_size 128 \
  --gpus 1

# get predicted sentences for predicted
python '../../postprocess_claim_verification.py' \
  --data_file "${data_dir}/train.jsonl" \
  --pred_sent_file "${pred_sent_dir}/train.jsonl" \
  --pred_claim_file "${out_dir}/${split}.out" \
  --out_file "${out_dir}/${split}.jsonl"

python '../../eval_fever.py' \
  --gold_file "${data_dir}/train.jsonl" \
  --pred_file "${out_dir}/${split}.jsonl" \
  --out_file "${out_dir}/eval.${split}.txt"

python '../../upsample_errors.py' \
  --gold_file "${data_dir}/train.jsonl" \
  --pred_file "${out_dir}/${split}.jsonl" \
  --upsample_rate 2 \
  --out_file "${out_dir}/upsampled.jsonl"
