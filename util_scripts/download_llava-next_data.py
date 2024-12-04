# https://huggingface.co/datasets/lmms-lab/LLaVA-NeXT-Data

import os
from datasets import load_dataset
from tqdm import tqdm
import json
from clean_data_json import clean_data

# Retry Loop
data = None
while data is None:
    try:
        data = load_dataset("lmms-lab/LLaVA-NeXT-Data", split="train")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        continue

image_folder = "playground/data/llava_next_images"
# Create the folder if it does not exist
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

json_path = "playground/data/llava_next_data.json"
converted_data = []
for da in tqdm(data):
    json_data = {}
    json_data["id"] = da["id"]
    if da["image"] is not None:
        # replace / with - in the image name to avoid FileNotFoundError: [Errno 2] No such file or directory: 'playground/data/llava_next_images/VG_100K/4.jpg'
        json_data["image"] = f"{da['id']}.jpg".replace('/', '-')
        try:
            da["image"].save(os.path.join(image_folder, json_data["image"]))
        except Exception as e:
        #convert to RGB format to save PNG files to RGB format to avoid error https://blog.csdn.net/weixin_41010198/article/details/87200236
            print(f"Error saving PNG File as JPG: {e}, convert to RGB format to save")
            da["image"].convert("RGB").save(os.path.join(image_folder, json_data["image"]))
    json_data["conversations"] = da["conversations"]
    converted_data.append(json_data)

converted_data = clean_data(converted_data)

# save the json
with open(json_path, "w") as f:
    json.dump(converted_data, f, indent=4, ensure_ascii=False)
