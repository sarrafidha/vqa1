import torch
import argparse
import json
from sklearn.metrics import accuracy_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def main(args):
    references = []
    with open(args.reference_path, "r") as j:
        gt_raw = json.load(j)
    gt_raw = [x for x in gt_raw if x["filepath"] == "test"]
    references = [x["changeflag"] for x in gt_raw]

    # answer
    hypotheses = []
    with open(args.answer_path, "r") as file:
        for line in file:
            if args.answer_path.endswith(".jsonl"):
                # 使用json.loads将JSON字符串解析为Python对象
                entry = json.loads(line)
                answer = entry["answer"].replace(".", "").strip()
            else:
                answer = line.strip()

            answer = 1 if answer == "Yes" else 0
            hypotheses.append(answer)

    assert len(references) == len(hypotheses)

    accuracy = accuracy_score(references, hypotheses)
    recall = recall_score(references, hypotheses)

    for gt, a, b in zip(gt_raw, references, hypotheses):
        if a != b:
            print(gt)

    print(f"分类准确率: {accuracy:.2%}")
    print(f"召回率: {recall:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        default=r"/root/autodl-tmp/LEVIR-MCI-dataset/LevirCCcaptions-v2.json",
    )
    parser.add_argument(
        "--answer_path",
        "-a",
        default="experiments/changechat-lora-gpt54k/results/ans_20240805_yes_or_no.jsonl",
    )
    args = parser.parse_args()
    main(args)
