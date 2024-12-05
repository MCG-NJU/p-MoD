# llava inference demo from llava README.md
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

model_path = "MCG-NJU/p-MoD-LLaVA-NeXT-7B"

prompt = "What are the things I should be cautious about when I visit here?"
image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "use_flash_attn": False,
    "max_new_tokens": 1000,
})()

eval_model(args)
