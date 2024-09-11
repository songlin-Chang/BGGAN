import os, sys
import os.path as osp
import time
import random
import datetime
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import clip as clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from open_clip import tokenize, create_model_and_transforms
from lib.utils import mkdir_p, get_rank, load_model_weights

from models.model import NetGTI, NetGIT, D_GET_SOCRE,echoCLIP_IMG_ENCODER

def prepare_models(args):
    device = args.device
    local_rank = args.local_rank
    multi_gpus = args.multi_gpus
    echoCLIP = clip.load('ViT-B/32')[0]
    echo_clip, _, preprocess_val = create_model_and_transforms(
    "hf-hub:mkaichristensen/echo-clip", precision="bf16", device="cuda"
)
    image_encoder = echoCLIP_IMG_ENCODER(echoCLIP).to(device)
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()
    # image_encoder = echo_clip

    text_encoder = echo_clip
    
    netGTI = NetGTI(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size).to(device)
    netGIT = NetGIT(args.nf).to(device)
    netD = D_GET_SOCRE(args.nf).to(device)
    return image_encoder, text_encoder, netGTI, netGIT, netD




