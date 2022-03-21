import jsonlines
import sys


def main(upsampled_file):
    class_count = {"N": 0, "R": 0, "S": 0}
    upsampled = [line for line in jsonlines.open(upsampled_file)]
    for example in upsampled:
        class_count[example["label"][0]] += 1
    print(class_count)


if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
