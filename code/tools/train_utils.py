import os
import random
from argparse import Namespace
import argparse
import json
import numpy as np
import torch
from open_clip import tokenize, create_model_and_transforms
from models.DAMSM import CNN_ENCODER

def parse_arguments() -> Namespace:
    parser = argparse.ArgumentParser(
        description="LT2326 H21 Mohamed's Project")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/data/coding/bggan/data/xray_h5",
                        help="Directory contains processed MS COCO dataset.")

    parser.add_argument(
        "--config_path",
        type=str,
        default="/data/coding/bggan/code/cfg/config.json",  # noqa: E501
        help="Path for the configuration json file.")

    parser.add_argument(
        '--device',
        type=str,
        default="gpu",  # gpu, cpu
        help='Device to be used either gpu or cpu.')

    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help='checkpoint filename.')

    args = parser.parse_args()

    return args


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(json_path: str) -> dict:
    with open(json_path) as json_file:
        data = json.load(json_file)

    return data

def prepare_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     echo_clip, _, preprocess_val = create_model_and_transforms(
#     "hf-hub:mkaichristensen/echo-clip", precision="bf16", device="cuda"
# )
#     # image encoder
#     image_encoder = echo_clip

    image_encoder = CNN_ENCODER(256)
    # 解码路径
    img_encoder_path = r"/data/coding/bggan/code/premodel/image_encoderx550.pth"
    state_dict = torch.load(img_encoder_path, map_location='cpu')
    image_encoder = load_model_weights(image_encoder, state_dict, multi_gpus=False)

    image_encoder.to(device)
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()
    return image_encoder


def load_netGIT(netGIT, path, multi_gpus, train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location="cpu")
    netGIT = load_model_weights(netGIT, checkpoint['model']['netGIT'], multi_gpus, train)
    netGIT.to(device)
    return netGIT

def load_model_weights(model, weights, multi_gpus, train=True):
    if list(weights.keys())[0].find('module')==-1:
        pretrained_with_multi_gpu = False
    else:
        pretrained_with_multi_gpu = True
    if (multi_gpus==False) or (train==False):
        if pretrained_with_multi_gpu:
            state_dict = {
                key[7:]: value
                for key, value in weights.items()
            }
        else:
            state_dict = weights
    else:
        state_dict = weights
    model.load_state_dict(state_dict)
    return model