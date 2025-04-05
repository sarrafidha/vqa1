import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import datetime
from changechat.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from changechat.conversation import conv_templates, SeparatorStyle
from changechat.model.builder import load_pretrained_model
from changechat.utils import disable_torch_init
from changechat.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    KeywordsStoppingCriteria,
)

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print(model_name)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name
    )

    questions = []
    questions = [
        json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
    ]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    answers_file = os.path.join(args.model_path, "results", f"ans_{now}.jsonl")
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    ans_file = open(answers_file, "w")

    for i in tqdm(range(0, len(questions), args.batch_size)):
        input_batch = []
        input_image_batch = []
        count = i
        image_folder = []
        batch_end = min(i + args.batch_size, len(questions))

        for j in range(i, batch_end):
            image_file = questions[j]["image"]
            # qs = questions[j]["text"]
            # qs = "Please judge whether these two images have changed. Please answer yes or no."
            qs = "Please briefly describe the changes in these two images. "
            qs = (
                DEFAULT_IMAGE_TOKEN
                + "\n"
                + DEFAULT_IMAGE_TOKEN
                + "\n"
                + qs
                # + "Please focus mainly on newly built or removed roads and houses, and ignore grass and vegetation. If only vegetation has changed or the objects that have changed are small, answer no change."
            )
            # qs = DEFAULT_IMAGE_TOKEN + '\n' + DEFAULT_IMAGE_TOKEN + '\n\n' + 'Please first determine whether the buildings and roads have changed, then count the number of changes, and finally describe the details of the changes in one sentence based on the above conclusions. Just tell me the details of the changes.'

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )
            input_batch.append(input_ids)

            if isinstance(image_file, str):
                image_file = [image_file]

            for image_f in image_file:
                image = Image.open(os.path.join(args.image_folder, image_f))
                image_folder.append(image)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

        max_length = max(tensor.size(1) for tensor in input_batch)

        final_input_list = [
            torch.cat(
                (
                    torch.zeros(
                        (1, max_length - tensor.size(1)),
                        dtype=tensor.dtype,
                        device=tensor.get_device(),
                    ),
                    tensor,
                ),
                dim=1,
            )
            for tensor in input_batch
        ]
        final_input_tensors = torch.cat(final_input_list, dim=0)
        image_tensor_batch = image_processor.preprocess(
            image_folder,
            crop_size={"height": 252, "width": 252},
            size={"shortest_edge": 252},
            return_tensors="pt",
        )["pixel_values"]

        with torch.inference_mode():
            output_ids = model.generate(
                final_input_tensors,
                images=image_tensor_batch.half().cuda(),
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=1,
                max_new_tokens=100,
                length_penalty=1,
                use_cache=True,
            )

        input_token_len = final_input_tensors.shape[1]
        n_diff_input_output = (
            (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
        )
        if n_diff_input_output > 0:
            print(
                f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
            )
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )
        for k in range(0, len(final_input_list)):
            output = outputs[k].strip()
            if output.endswith(stop_str):
                output = output[: -len(stop_str)]
            output = output.strip()

            ans_id = shortuuid.uuid()

            ans_file.write(
                json.dumps(
                    {
                        "question_id": questions[count]["question_id"],
                        "image_id": questions[count]["image"],
                        "answer": output,
                    }
                )
                + "\n"
            )
            count = count + 1
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, default="experiments/changechat-lora-gpt-114k"
    )
    parser.add_argument("--model-base", type=str, default="hf-models/llavav1.5-7b")
    parser.add_argument(
        "--image-folder", type=str, default="/root/autodl-tmp/LEVIR-MCI-dataset/images"
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default="/root/autodl-tmp/LEVIR-MCI-dataset/Test_CC_v3.jsonl",
    )
    parser.add_argument("--answers-file", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")  # llava_v1
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    eval_model(args)
