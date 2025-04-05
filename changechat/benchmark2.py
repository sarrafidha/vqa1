import torch
from transformers import GPT2Tokenizer
import argparse
import time
import json
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def get_eval_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]

    hypo = [
        [" ".join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]
    ]
    ref = [
        [" ".join(reft) for reft in reftmp]
        for reftmp in [
            [[str(x) for x in reft] for reft in reftmp] for reftmp in references
        ]
    ]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        (
            method.extend(method_i)
            if isinstance(method_i, list)
            else method.append(method_i)
        )
        # print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict


def pad_tokens(tokenizer, caption):
    tokens = torch.tensor(tokenizer.encode(caption))
    padding = 50 - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        # self.captions[item] = tokens
    elif padding < 0:
        tokens = tokens[:50]
        # self.captions[item] = tokens
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = 0
    return tokens


def main(args):
    # tokenizer
    gpt2_type = "/root/autodl-tmp/gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    # end_value = tokenizer.encode('.')[0]

    # references
    references_all = []
    references_no_changed = []
    references_changed = []
    with open(args.reference_path, "r") as j:
        gt_raw = json.load(j)
    gt_raw = [x for x in gt_raw if x["filepath"] == "test"]
    for item in gt_raw:
        caption = []
        # item["sentences"] = [
        #     x["raw"].replace(".", "").strip() for x in item["sentences"]
        # ]
        for sentence in item["sentences"]:
            answer = torch.tensor(tokenizer.encode(sentence)).tolist()
            caption.append(answer)
        if item["changeflag"] != item["changeflag2"]:
            continue

        if item["changeflag"] == 0:
            references_no_changed.append(caption)
        else:
            references_changed.append(caption)
        references_all.append(caption)

    # answer
    hypotheses_all = []
    hypotheses_no_changed = []
    hypotheses_changed = []

    with open(args.answer_path, "r") as file:
        for gt, line in zip(gt_raw, file):
            if args.answer_path.endswith(".jsonl"):
                # 使用json.loads将JSON字符串解析为Python对象
                entry = json.loads(line)
                answer = entry["answer"].replace(".", "").strip().lower()
            else:
                answer = line.strip()

            # # 处理entry，例如打印
            # answer = pad_tokens(tokenizer, answer).tolist()
            # if end_value in answer:
            #     end_index = answer.index(end_value)
            #     ref_remo = answer[:end_index]
            # else:
            #     while answer and answer[-1] == 0:
            #         answer.pop()
            #     ref_remo = answer
            #     # print(ref_remo)

            answer = torch.tensor(tokenizer.encode(answer)).tolist()
            if gt["changeflag"] != gt["changeflag2"]:
                print("drop")
                continue
            if gt["changeflag"] == 0:
                hypotheses_no_changed.append(answer)
            else:
                hypotheses_changed.append(answer)

            hypotheses_all.append(answer)

    assert len(references_all) == len(hypotheses_all)
    assert len(references_no_changed) == len(hypotheses_no_changed)
    assert len(references_changed) == len(hypotheses_changed)

    print("====== no changed results ======")
    metrics_no_changed = get_eval_score(references_no_changed, hypotheses_no_changed)
    print(
        "BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format(
            metrics_no_changed["Bleu_1"],
            metrics_no_changed["Bleu_2"],
            metrics_no_changed["Bleu_3"],
            metrics_no_changed["Bleu_4"],
            metrics_no_changed["METEOR"],
            metrics_no_changed["ROUGE_L"],
            metrics_no_changed["CIDEr"],
        )
    )
    print("\n")
    print("====== changed results ======")
    metrics_changed = get_eval_score(references_changed, hypotheses_changed)
    print(
        "BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format(
            metrics_changed["Bleu_1"],
            metrics_changed["Bleu_2"],
            metrics_changed["Bleu_3"],
            metrics_changed["Bleu_4"],
            metrics_changed["METEOR"],
            metrics_changed["ROUGE_L"],
            metrics_changed["CIDEr"],
        )
    )
    print("\n")
    print("====== all results ======")
    metrics_all = get_eval_score(references_all, hypotheses_all)
    print(
        "BLEU-1 {} BLEU-2 {} BLEU-3 {} BLEU-4 {} METEOR {} ROUGE_L {} CIDEr {}".format(
            metrics_all["Bleu_1"],
            metrics_all["Bleu_2"],
            metrics_all["Bleu_3"],
            metrics_all["Bleu_4"],
            metrics_all["METEOR"],
            metrics_all["ROUGE_L"],
            metrics_all["CIDEr"],
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        default=r"/root/autodl-tmp/LEVIR-MCI-dataset/LevirCCcaptions-v2.json",
    )
    parser.add_argument("--answer_path", "-a")
    args = parser.parse_args()
    main(args)
