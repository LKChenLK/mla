#!/bin/bash
#
# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Authors: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
#SBATCH --job-name=predict_roberta-large
#SBATCH --out='predict_roberta-large.log'
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:tesla_a100:1

conda_setup="/home/smg/$(whoami)/miniconda3/etc/profile.d/conda.sh"
if [[ -f "${conda_setup}" ]]; then
  #shellcheck disable=SC1090
  . "${conda_setup}"
  conda activate mla
fi

set -ex

pretrained='roberta-large'
max_len=128
model_dir="${pretrained}-${max_len}-mod"
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

split='dev'

if [[ -f "${out_dir}/${split}.jsonl" ]]; then
  echo "Result '${out_dir}/${split}.jsonl' exists!"
  exit
fi

python '../../preprocess_covidfact.py' \
  --in_file "${data_dir}/${split}_covidfact.tsv" \
  --out_file "${out_dir}/${split}.tsv" \
  --max_evidence_per_claim 5

python '../../predict_covidfact.py' \
  --checkpoint_file "${latest}" \
  --in_file "${out_dir}/${split}.tsv" \
  --out_file "${out_dir}/${split}.out" \
  --batch_size 128 \
  --gpus 1

python '../../postprocess_covidfact.py' \
  --data_file "${data_dir}/${split}_covidfact.tsv" \
  --pred_claim_file "${out_dir}/${split}.out" \
  --out_file "${out_dir}/${split}.jsonl"

# note: no sentence selection for covidfact dataset
python '../../eval_covidfact.py' \
  --gold_file "${data_dir}/${split}_covidfact.tsv" \
  --pred_file "${out_dir}/${split}.jsonl" \
  --out_file "${out_dir}/eval.${split}.txt"
