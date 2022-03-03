#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=retrain_bert-base
#SBATCH --out='retrain_bert-base.log'
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex

task='claim-verification'
pretrained='bert-base-uncased'
max_len=128
model_dir="${pretrained}-${max_len}-mod"
inp_dir="${pretrained}-${max_len}-inp"
out_dir="${pretrained}-${max_len}-out"

data_dir='../data'

model='verification-joint'
aggregate_mode='attn'
attn_bias_type='value_only'

if [[ -d "${model_dir}" ]]; then
  echo "${model_dir} exists! Skip training."
  exit
fi

mkdir -p "${inp_dir}"

python '../../preprocess_claim_verification.py' \
  --corpus "${data_dir}/corpus.jsonl" \
  --in_file "${out_dir}/upsampled.jsonl" \
  --out_file "${inp_dir}/train.tsv" \
  --training

python '../../train.py' \
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
  --use_title \
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
  --gpus 1 \
  --overwrite_cache