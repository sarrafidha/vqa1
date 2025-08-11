import json
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch
import os

# Chemins à adapter
model_id = "llava-hf/llava-1.5-7b-hf"
data_path = "C:/Users/sarra/Downloads/changechat/train/train/json/Train_50images.json"
image_folder = "C:/Users/sarra/Downloads/changechat/train/train/image"

# Charger le processor et le modèle LLaVA
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# Charger les données de test
with open(data_path, "r") as f:
    data = json.load(f)["question"]

# Tester sur les 5 premiers exemples
for sample in data[:5]:
    img_id = sample["img_id"]
    im1_path = os.path.join(image_folder, "im1", img_id)
    question = sample["question"]

    image = Image.open(im1_path).convert("RGB")
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(prompt, image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
        answer = processor.decode(output[0], skip_special_tokens=True)
    print(f"Q: {question}")
    print(f"Réponse LLaVA: {answer}")
    print(f"Réponse attendue: {sample['answer']}")
    print('-'*40) 