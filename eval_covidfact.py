# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.

import argparse
import jsonlines
import csv
import io
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

label_equiv = {"entailment": "SUPPORTS", "not_entailment": "REFUTES"}


def label_accuracy(pred_labels, gold_labels):
    correct = 0
    total = 0

    # for no-NEI evaluation, non-NEI labels from actual_labels (gold_preds)
    for idx, pred_label in enumerate(pred_labels):
        if pred_label == gold_labels[idx]:
            correct += 1.0

    total = len(pred_labels)
    acc = correct / total

    return acc


def main(pred_file, gold_file):
    gold = list(csv.reader(open(gold_file, "r", encoding="utf-8-sig"), delimiter="\t"))
    pred = [line for line in jsonlines.open(pred_file)]
    assert len(pred) == len(gold)
    gold_labels = [line[3] for line in gold]
    pred_labels = [line["predicted_label"] for line in pred]

    label_acc = label_accuracy(pred_labels, gold_labels)
    pred_equiv_labels = [label_equiv[label] for label in pred_labels]
    label_count = Counter(pred_equiv_labels)
    res = "\n".join(
        [
            "           [S     R     ]",
            f"Precision: {precision_score(gold_labels, pred_labels, average=None).round(4)*100.0}",
            f"Recall:    {recall_score(gold_labels, pred_labels, average=None).round(4)*100.0}",
            f"F1:        {f1_score(gold_labels, pred_labels, average=None).round(4)*100.0}",
            "",
            "Confusion Matrix:",
            f"{confusion_matrix(gold_labels, pred_labels)}",
            "",
            f"{label_count}",
            "",
            f"Label accuracy:     {label_acc*100.0:.4}",
        ]
    )
    print(res)
    print(f"Save to {args.out_file}")
    with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
        out.write(res + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()
    main(args.pred_file, args.gold_file)
