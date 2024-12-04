import argparse
import random
import json
from tqdm import tqdm

def clean_data(data_json):
    # Fix random seed
    random.seed(42)

    # 1. Fix some broken data that will cause "tokenization mismatch" error
    sep = 'ASSISTANT: '
    for sources in tqdm(data_json):
        for conversation in sources['conversations']:
            if sep in conversation['value']:
                conversation['value'] = conversation['value'].replace(sep, '')

    # 2. Make sure length of multi-modal and text-only data are both divisible by 128
    with_image = [d for d in data_json if 'image' in d]
    # Calculate the number of elements to append
    num_elements = 128 - (len(with_image) % 128)
    print(f"Number of elements to append to multimodal data: {num_elements}")
    # Randomly select elements from the list
    random_elements = random.sample(with_image, num_elements)
    # Append the random elements to the end of the list
    data_json += random_elements

    without_image = [d for d in data_json if 'image' not in d]
    # Calculate the number of elements to append
    num_elements = 128 - (len(without_image) % 128)
    print(f"Number of elements to append to text-only data: {num_elements}")
    # Randomly select elements from the list
    random_elements = random.sample(without_image, num_elements)
    # Append the random elements to the end of the list
    data_json += random_elements

    return data_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_json_path", type=str, required=True)
    parser.add_argument("--cleaned_json_path", type=str, required=True)

    args = parser.parse_args()

    data_json = json.load(open(args.original_json_path, "r"))
    data_json = clean_data(data_json)
    with open(args.cleaned_json_path, "w") as f:
        json.dump(data_json, f, indent=4, ensure_ascii=False)


