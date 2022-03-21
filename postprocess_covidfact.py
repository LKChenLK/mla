# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import csv
import jsonlines
import numpy as np
from preprocess_covidfact import split_dict, get_split_claim_id


def get_predictions(data_file, pred_claim_file):
    """Gets predicted veracity class, save it with selected evidence"""
    lines_0 = list(
        csv.reader(open(data_file, "r", encoding="utf-8-sig"), delimiter="\t")
    )
    lines_1 = list(
        csv.reader(open(pred_claim_file, "r", encoding="utf-8-sig"), delimiter=" ")
    )
    assert len(lines_0) == len(lines_1)

    labels = ["entailment", "not_entailment"]
    predictions = {}
    for line_0, line_1 in zip(lines_0, lines_1):
        scores = np.asarray(list(map(float, line_1)))
        label_idx = np.argmax(scores)
        label = labels[label_idx]

        index = int(line_0[0])
        claim_id = get_split_claim_id(index, split_dict, pred_claim_file)
        evidence = line_0[1]
        predictions[claim_id] = {"predicted_label": label, "evidence": evidence}
    return predictions


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--pred_claim_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    return parser.parse_args()


def main():
    args = build_args()
    predictions = get_predictions(args.data_file, args.pred_claim_file)

    print(f"Save to {args.out_file}")
    # Save outfile in the same order of data_file (for eval)

    fin = list(
        csv.reader(open(args.data_file, "r", encoding="utf-8-sig"), delimiter="\t")
    )
    with jsonlines.open(args.out_file, "w") as out:
        for line in fin:
            index = line[0]
            claim_id = get_split_claim_id(index, split_dict, args.data_file)
            out.write(
                {
                    "id": claim_id,
                    "predicted_label": predictions[claim_id]["predicted_label"],
                    "predicted_evidence": predictions[claim_id]["evidence"],
                }
            )


if __name__ == "__main__":
    main()
