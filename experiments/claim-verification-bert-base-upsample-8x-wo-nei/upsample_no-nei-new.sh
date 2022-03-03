#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=upsample_bert-base
#SBATCH --out='upsample_nonei.log'
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex

pretrained="bert-base-uncased"
max_len=128
out_dir="${pretrained}-${max_len}-out"
data_dir='../data'
split='first_train'
out_name='upsampled_nonei'

mkdir -p "${out_dir}"

if [[ -f "${out_dir}/${out_name}.jsonl" ]]; then
  echo "Result '${out_dir}/${out_name}.jsonl' exists!"
  exit
fi

python '../../upsample_errors.py' \
  --gold_file "${data_dir}/train.jsonl" \
  --pred_file "${out_dir}/${split}.jsonl" \
  --upsample_rate 8 \
  --upsample_error_type "no_nei" \
  --upsample_error_rate 1 \
  --out_file "${out_dir}/${out_name}.jsonl"

