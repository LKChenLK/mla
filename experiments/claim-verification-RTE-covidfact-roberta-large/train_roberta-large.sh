#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=train_roberta-large
#SBATCH --out='train_roberta-large.log'
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex

task='claim-verification'
pretrained='roberta-large'
max_len=128
model_dir="${pretrained}-${max_len}-mod"
inp_dir="${pretrained}-${max_len}-inp"

data_dir='../data'

model='verification'
aggregate_mode='attn'
attn_bias_type='value_only'

if [[ -d "${model_dir}" ]]; then
  echo "${model_dir} exists! Skip training."
  exit
fi

mkdir -p "${inp_dir}"

python '../../preprocess_covidfact.py' \
  --in_file "${data_dir}/train_covidfact.tsv" \
  --out_file "${inp_dir}/train.tsv" \
  --training \
  --max_evidence_per_claim 5

python '../../train_covidfact.py' \
  --task "${task}" \
  --data_dir "${inp_dir}" \
  --default_root_dir "${model_dir}" \
  --pretrained_model_name "${pretrained}" \
  --max_seq_length "${max_len}" \
  --model_name "${model}" \
  --aggregate_mode "${aggregate_mode}" \
  --attn_bias_type "${attn_bias_type}" \
  --sent_attn \
  --word_attn \
  --class_weighting \
  --max_epochs 2 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  --accumulate_grad_batches 16 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.06 \
  --adafactor \
  --gradient_clip_val 1.0 \
  --precision 16 \
  --deterministic true \
  --gpus 1