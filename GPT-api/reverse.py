import json, re
from openai import OpenAI
import concurrent.futures
from time import sleep
import ast, tqdm

# set the key
API_SECRET_KEY = "sk-zk2277dd97db0e3d8a2600807992f389c5365b8fb3005750"
BASE_URL = "https://flag.smarttrot.com/v1/"


def chat_completions3(query):
    try:
        client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ],
        )
        return resp.choices[0].message.content
    except:
        return "ERROR"


def getExistOutputs():
    with open("output.txt", "r") as file:
        exist_outputs = [ast.literal_eval(line.strip()) for line in file]
    exist_outputs = [x[1] for x in exist_outputs]
    return exist_outputs


def getReverseSentence(imgid, sentid, changeflag, input):
    print(sentid)
    # 有变化使用chatgpt生成，无变化直接返回原输入
    if not changeflag:
        return imgid, sentid, input

    template = "If the change from image A to image B is described as: '{0}', then the inverse change from B to A is described as:"
    pattern = r'["\']([^"\']*)["\']'

    rslt = chat_completions3(template.format(input))
    matches = re.findall(pattern, rslt)
    if matches:
        rslt = matches[0]
    return imgid, sentid, rslt


def write_to_file(result):
    output_file = "output.txt"
    with open(output_file, "a") as f:
        f.write(result + "\n")


with open("LevirCCcaptions.json", "r") as f:
    datajson = json.load(f)
    images = datajson["images"]


if __name__ == "__main__":
    inputs = []
    exist_outputs = getExistOutputs()

    for im in images:
        for sen in im["sentences"]:
            inputs.append([sen["imgid"], sen["sentid"], im["changeflag"], sen["raw"]])
    inputs = list(filter(lambda x: x[1] not in exist_outputs, inputs))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 并行
        results = executor.map(lambda x: getReverseSentence(*x), inputs)
        # 将结果写入文件
        for result in results:
            if result:
                write_to_file(str(result))
