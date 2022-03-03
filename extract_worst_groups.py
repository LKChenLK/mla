import argparse
import jsonlines
import io
from tqdm import tqdm
from fever_scorer import is_correct_label
from preprocess_claim_verification import get_all_sentences, PAD_SENT


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


def get_worst_group(pred, gold):
    incorrect_inst = []
    spurious_correlated_inst = []
    neg_ref_inst = []

    for idx, instance in enumerate(pred):
        assert (
            "evidence" in gold[idx].keys()
        ), "evidence must be provided for the actual evidence"
        instance["evidence"] = gold[idx]["evidence"]
        instance["label"] = gold[idx]["label"]
        instance["claim"] = gold[idx]["claim"]

        for p in instance["predicted_evidence"]:
            p.append(0)

        claim = instance["claim"]
        if not is_correct_label(instance):
            incorrect_inst.append(instance)
            if has_negation_word(claim) and instance["label"][0] == "R":
                spurious_correlated_inst.append(instance)

        if has_negation_word(claim) and instance["label"][0] == "R":
            neg_ref_inst.append(instance)

    assert args.worst_group_type in [
        "all_incorrects",
        "spurious_refutes",
        "negation_refutes",
    ]
    if args.worst_group_type == "all_incorrects":
        out_inst = incorrect_inst
    elif args.worst_group_type == "spurious_refutes":
        out_inst = spurious_correlated_inst
    elif args.worst_group_type == "negation_refutes":
        out_inst = neg_ref_inst
    return out_inst


def build_examples(corpus, line):
    claim_id = line["id"]
    claim_text = line["claim"]
    pred_evidence = line["predicted_evidence"]
    examples = []

    examples.append([claim_id, claim_text] + PAD_SENT)
    for evidence_sent in get_all_sentences(
        corpus, pred_evidence, args.max_evidence_per_claim
    ):
        examples.append([claim_id, claim_text] + evidence_sent)

    return examples


def main(pred_file, gold_file):

    gold = [line for line in jsonlines.open(gold_file)]
    pred = [line for line in jsonlines.open(pred_file)]
    assert len(pred) == len(gold)

    # get sentences for training
    out_inst = get_worst_group(pred, gold)
    corpus = {doc["doc_id"]: doc for doc in jsonlines.open(args.corpus)}
    out_examples = []

    for instance in tqdm(
        out_inst, total=len(out_inst), desc="Building worst-group examples"
    ):
        out_examples.extend(build_examples(corpus, instance))

    # tsv file for training
    print(f"Save worst group to {args.out_pred_file}")
    with io.open(args.out_pred_file, "w", encoding="utf-8", errors="ignore") as out:
        for e in out_examples:
            e = list(map(str, e))
            out.write("\t".join(e) + "\n")

    # jsonl file for postprocessing and eval after predict
    print(
        f"Save worst group sentences to {args.out_pred_sent_file}, answers to {args.out_gold_file}"
    )
    with jsonlines.open(args.out_pred_sent_file, "w") as pred_sent_out, jsonlines.open(
        args.out_gold_file, "w"
    ) as gold_out:
        for instance in out_inst:
            claim_id = instance["id"]
            label = instance["label"]
            claim = instance["claim"]
            pred_evidence = instance["predicted_evidence"]
            evidence = instance["evidence"]

            pred_sent_out.write(
                {
                    "id": claim_id,
                    "label": label,
                    "claim": claim,
                    "evidence": evidence,
                    "predicted_evidence": pred_evidence,
                }
            )

            gold_out.write(
                {
                    "id": claim_id,
                    "label": label,
                    "claim": claim,
                    "evidence": evidence,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_pred_file", type=str, required=True)
    parser.add_argument("--out_gold_file", type=str, required=True)
    parser.add_argument("--out_pred_sent_file", type=str, required=True)
    parser.add_argument("--max_evidence_per_claim", type=int, default=5)
    parser.add_argument("--worst_group_type", type=str, required=True)

    args = parser.parse_args()
    main(args.pred_file, args.gold_file)
