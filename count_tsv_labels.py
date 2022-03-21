import sys
import csv


def main(in_file):
    class_count = {"R": 0, "S": 0}
    label_dict = {"entailment": "S", "not_entailment": "R"}
    lines = list(csv.reader(open(in_file, "r", encoding="utf-8-sig"), delimiter="\t"))
    for example in lines:
        class_count[label_dict[example[3]]] += 1
    print(class_count)


if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
