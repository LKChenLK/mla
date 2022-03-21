import argparse
import jsonlines
from fever_scorer import is_correct_label


def has_negation_word(claim):
    negation_set = set(
        [
            "not",
            "yet",
            "refuse",
            "refused",
            "fail",
            "failed",
            "only",
            "incapable",
            "unable",
            "no",
            "neither",
            "never",
            "none",
        ]
    )
    for word in claim.split():
        word = word.lower().strip(".").strip(",")
        if word in negation_set:
            return True
    return False


def main(pred_file, gold_file):
    # get predicted claims and sents
    # find erroneous preds
    # format then as same format for training

    gold = [line for line in jsonlines.open(gold_file)]
    pred = [line for line in jsonlines.open(pred_file)]
    gold_dict_all = {line["id"]: line for line in gold}
    gold_dict = {line["id"]: gold_dict_all[line["id"]] for line in pred}

    incorrect_inst = []
    extra_error_inst = []
    correct_inst = []
    for idx, instance in enumerate(pred):
        claim_id = instance["id"]
        assert (
            "evidence" in gold_dict[claim_id].keys()
        ), "evidence must be provided for the actual evidence"

        instance["evidence"] = gold_dict[claim_id]["evidence"]
        instance["label"] = gold_dict[claim_id]["label"]
        instance["claim"] = gold_dict[claim_id]["claim"]
        for p in instance["predicted_evidence"]:
            p.append(0)  # need dummy score as input format;
            # evd sentences are already sorted by score

        if is_correct_label(instance):
            correct_inst.append(instance)
        else:
            if args.upsample_error_type and args.upsample_error_type == "no_nei":
                if instance["label"][0] in {"R", "S"}:
                    incorrect_inst.append(instance)
            else:
                incorrect_inst.append(instance)

        claim = instance["claim"]
        if args.upsample_error_type:
            assert args.upsample_error_rate > 0
            if args.upsample_error_type == "negation_refutes":
                if has_negation_word(claim) and instance["label"][0] == "R":
                    extra_error_inst.append(instance)

    out_inst = correct_inst.copy()
    if args.upsample_error_type == "non_errors":
        out_inst.extend(correct_inst * (args.upsample_error_rate))
    else:  # no error type sp
        out_inst.extend(
            incorrect_inst * (args.upsample_rate + 1)
        )  # include original count

    if extra_error_inst:
        out_inst.extend(extra_error_inst * args.error_upsample_rate)

    print(f"Save to {args.out_file}")
    with jsonlines.open(args.out_file, "w") as out:
        for instance in out_inst:
            claim_id = instance["id"]
            label = instance["label"]
            claim = instance["claim"]
            pred_evidence = instance["predicted_evidence"]
            evidence = instance["evidence"]

            out.write(
                {
                    "id": claim_id,
                    "label": label,
                    "claim": claim,
                    "evidence": evidence,
                    "predicted_evidence": pred_evidence,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--upsample_rate", type=int, default=0)
    parser.add_argument("--upsample_error_type", type=str)
    parser.add_argument("--upsample_error_rate", type=int, default=0)
    parser.add_argument("--out_file", type=str, required=True)

    args = parser.parse_args()
    main(args.pred_file, args.gold_file)
