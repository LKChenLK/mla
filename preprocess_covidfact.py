import argparse
import io
from tqdm import tqdm

from spacy.lang.en import English

nlp = English()
nlp.add_pipe("sentencizer")

PAD_ID = -1
PAD_SENT = ["[PAD]"]
PAD_LABEL = -1


def pad_to_max(sent_list, max_evidence_per_claim):
    if len(sent_list) < max_evidence_per_claim:
        sent_list += PAD_SENT * (max_evidence_per_claim - len(sent_list))


def add_period(sentence):
    assert isinstance(sentence, str), "Input should be a string."
    if sentence[-1] != ".":
        sentence += "."
    return sentence


def get_sentences(evidences, max_evidence_per_claim):
    evidence_sentences = []
    evidence_sentences = nlp(evidences)
    evidence_sentences = [sent.text for sent in evidence_sentences.sents]
    pad_to_max(evidence_sentences, max_evidence_per_claim)
    evidence_sentences = evidence_sentences[:max_evidence_per_claim]

    assert (
        len(evidence_sentences) == max_evidence_per_claim
    ), f"max_evidence_per_claim: {max_evidence_per_claim}, \
            len(evidence_sentences): {len(evidence_sentences)}"

    for evd in evidence_sentences:
        if isinstance(evd, str):
            add_period(evd)
        yield [evd]
        # original: [doc_id, sent_id, sent_text, label, selection_label]


def build_examples(args, line):
    example_items = line.strip().split("\t")
    index, evidences, claim, label = example_items
    selection_label = PAD_LABEL
    doc_id = PAD_SENT
    sent_id = PAD_ID

    add_period(claim)

    out_examples = []
    if args.training:
        out_examples.append(
            [index, claim] + doc_id + [sent_id] + PAD_SENT + [label, selection_label]
        )
        for evidence_sent in get_sentences(
            evidences,
            args.max_evidence_per_claim,
        ):
            out_examples.append(
                [index, claim]
                + doc_id
                + [sent_id]
                + evidence_sent
                + [label, selection_label]
            )
    else:
        out_examples.append([index, claim] + doc_id + [sent_id] + PAD_SENT)
        for evidence_sent in get_sentences(evidences, args.max_evidence_per_claim):
            out_examples.append([index, claim] + doc_id + [sent_id] + evidence_sent)
    return out_examples


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--max_evidence_per_claim", type=int, default=5)
    return parser.parse_args()


def main():
    args = build_args()
    with open(args.in_file) as fin:  # e.g. './RTE-covidfact/test1.tsv'
        infile = fin.readlines()

    out_examples = []

    for index, line in enumerate(
        tqdm(infile, total=len(infile), desc="Building examples")
    ):
        if index == 0:
            continue
        out_examples.extend(build_examples(args, line))

    print("Number of examples: ", len(out_examples))
    print(f"Save to {args.out_file}")
    with io.open(args.out_file, "w", encoding="utf-8", errors="ignore") as out:
        for e in out_examples:
            e = list(map(str, e))
            out.write("\t".join(e) + "\n")


if __name__ == "__main__":
    main()
