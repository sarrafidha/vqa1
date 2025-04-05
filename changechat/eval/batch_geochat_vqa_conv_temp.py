import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import datetime
from changechat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from changechat.conversation import conv_templates, SeparatorStyle
from changechat.model.builder import load_pretrained_model
from changechat.utils import disable_torch_init
from changechat.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math

QSList = [
    "Have the objects in these two remote sensing images changed? Please answer yes or no.",
    "How many changes have occurred to the buildings and roads in the two images?",
    "Based on the above analysis, please summarize the changes that have taken place in these two images."
]


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, "geochat_lora" + model_name)
    
    questions=[]
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    # answers_file = os.path.expanduser(args.answers_file)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    answers_file = os.path.join(args.model_path, 'results', f'ans_{now}.jsonl')
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    
    ans_file = open(answers_file, "w")
    
    for i in tqdm(range(0,len(questions),args.batch_size)):
        count=i
        image_folder=[]     
        batch_end = min(i + args.batch_size, len(questions))
        conv_history = [None] * args.batch_size
        for turn_idx in range(len(QSList)):
            input_batch=[]
            # collect for a batch
            for j in range(i,batch_end):
                # 第一轮对话加上图像标签
                if turn_idx == 0:
                    # 第一轮对话初始化
                    conv_history[j-i] = conv_templates[args.conv_mode].copy()
                    qs = DEFAULT_IMAGE_TOKEN + '\n' + DEFAULT_IMAGE_TOKEN + '\n' + QSList[turn_idx]

                    # 第一轮对话加载图像信息
                    image_file=questions[j]['image']
                    if isinstance(image_file, str):
                        image_file = [image_file]
                    for image_f in image_file:
                        image = Image.open(os.path.join(args.image_folder, image_f))
                        image_folder.append(image)
                else:
                    qs = QSList[turn_idx]
                
                conv_history[j-i].append_message(conv_history[j-i].roles[0], qs)
                conv_history[j-i].append_message(conv_history[j-i].roles[1], None)
                prompt = conv_history[j-i].get_prompt()

                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                input_batch.append(input_ids)
                stop_str = conv_history[j-i].sep if conv_history[j-i].sep_style != SeparatorStyle.TWO else conv_history[j-i].sep2

            max_length = max(tensor.size(1) for tensor in input_batch)
            final_input_list = [torch.cat((torch.zeros((1,max_length - tensor.size(1)), dtype=tensor.dtype,device=tensor.get_device()), tensor),dim=1) for tensor in input_batch]
            final_input_tensors=torch.cat(final_input_list,dim=0)
            if turn_idx == 0:
                image_tensor_batch = image_processor.preprocess(image_folder,crop_size ={'height': 252, 'width': 252},size = {'shortest_edge': 256}, return_tensors='pt')['pixel_values']

            with torch.inference_mode():
                output_ids = model.generate(final_input_tensors, images=image_tensor_batch.half().cuda(), do_sample=False , temperature=args.temperature, top_p=args.top_p, num_beams=1, max_new_tokens=100,length_penalty=1, use_cache=True)

            input_token_len = final_input_tensors.shape[1]
            n_diff_input_output = (final_input_tensors != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            for k in range(0,len(final_input_list)):
                output = outputs[k].strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip()
                conv_history[k].messages.pop()
                if turn_idx == 0:
                    output = "Yes" if questions[i+k]['changeflag'] == 1 else "No"
                    print(i+k,output)
                conv_history[k].append_message(conv_history[k].roles[1], output)
            
        for k in range(0, len(outputs)):
            ans_file.write(json.dumps({
                                "question_id": questions[count]["question_id"],
                                "image_id": questions[count]["image"],
                                "answer": conv_history[k].get_prompt(),
                                }) + "\n")
            count=count+1
            ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="experiments/geochat-cc-lora-0701")
    parser.add_argument("--model-base", type=str, default="hf-models/llavav1.5-7b")
    parser.add_argument("--image-folder", type=str, default="/root/autodl-tmp/LEVIR-MCI-dataset/images")
    parser.add_argument("--question-file", type=str, default="/root/autodl-tmp/LEVIR-MCI-dataset/Test_CC_v3.jsonl")
    parser.add_argument("--answers-file", type=str, default="/root/autodl-tmp/LEVIR-MCI-dataset/answer_v3_0701_a.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")  # llava_v1
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--batch_size",type=int, default=8)
    args = parser.parse_args()

    eval_model(args)
