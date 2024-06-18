import os

import errno
import numpy as np
import numpy.random as random
import torch
from torch import distributed as dist

import pickle

import yaml
from easydict import EasyDict as edict

import datetime
import dateutil.tz

from PIL import Image


# test_utils
def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def load_npz(path):
    f = np.load(path)
    m, s = f['mu'][:], f['sigma'][:]
    f.close()
    return m, s


def truncated_noise(batch_size=1, dim_z=100, truncation=1., seed=None):
    from scipy.stats import truncnorm
    state = None if seed is None else np.random.RandomState(seed)
    values = truncnorm.rvs(-2, 2, size=(batch_size, dim_z), random_state=state).astype(np.float32)
    return truncation * values


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_time_stamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')  
    return timestamp


def load_yaml(filename):
    with open(filename, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    return cfg


def merge_args_yaml(args):
    if args.cfg_file is not None:
        opt = vars(args)
        args = load_yaml(args.cfg_file)
        args.update(opt)
        args = edict(args)
    return args


def save_args(save_path, args):
    fp = open(save_path, 'w')
    fp.write(yaml.dump(args))
    fp.close()


# DDP utils
def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


# save and load models
def load_opt_weights(optimizer, weights):
    optimizer.load_state_dict(weights)
    return optimizer



def load_netGTI(netGTI, path, multi_gpus, train):
    checkpoint = torch.load(path, map_location="cpu")
    netGTI = load_model_weights(netGTI, checkpoint['model']['netGTI'], multi_gpus, train)
    return netGTI

def load_netGIT(netGIT, path, multi_gpus, train):
    checkpoint = torch.load(path, map_location="cpu")
    netGIT = load_model_weights(netGIT, checkpoint['model']['netGIT'], multi_gpus, train)
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


# data util
def write_to_txt(filename, contents): 
    fh = open(filename, 'w') 
    fh.write(contents) 
    fh.close()


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


###########  GEN  #############
def get_tokenizer():
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer


def tokenize(wordtoix, text_filepath):
    '''generate images from example sentences'''
    tokenizer = get_tokenizer()
    filepath = text_filepath
    with open(filepath, "r") as f:
        sentences = f.read().encode('utf-8').decode('utf8').split('\n')
        # a list of indices for a sentence
        captions = []
        cap_lens = []
        new_sent = []
        for sent in sentences:
            if len(sent) == 0:
                continue
            sent = sent.replace("\ufffd\ufffd", " ")
            # tokens = tokenizer.tokenize(sent.lower())
            tokens = tokenizer.tokenize(sent)
            if len(tokens) == 0:
                print('sent', sent)
                continue
            rev = []
            for t in tokens:
                # t = t.encode('ascii', 'ignore').decode('ascii')
                if len(t) > 0 and t in wordtoix:
                    rev.append(wordtoix[t])
            captions.append(rev)
            cap_lens.append(len(rev))
            new_sent.append(sent)
        return captions, cap_lens, new_sent


def sort_example_captions(captions, cap_lens, device):
    max_len = np.max(cap_lens)
    sorted_indices = np.argsort(cap_lens)[::-1]
    cap_lens = np.asarray(cap_lens)
    cap_lens = cap_lens[sorted_indices]
    cap_array = np.zeros((len(captions), max_len), dtype='int64')
    for i in range(len(captions)):
        idx = sorted_indices[i]
        cap = captions[idx]
        c_len = len(cap)
        cap_array[i, :c_len] = cap
    captions = torch.from_numpy(cap_array).to(device)
    cap_lens = torch.from_numpy(cap_lens).to(device)
    return captions, cap_lens, sorted_indices


def prepare_sample_data(captions, caption_lens, text_encoder, device):
    print('*'*40)
    captions, sorted_cap_lens, sorted_cap_idxs = sort_example_captions(captions, caption_lens, device)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    return sent_emb, words_embs


def encode_tokens(text_encoder, caption, cap_lens):
    # encode text
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))
        else:
            hidden = text_encoder.init_hidden(caption.size(0))

        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs 


def sort_sents(captions, caption_lens, device):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)
    captions = captions[sorted_cap_indices].squeeze()
    captions = captions.to(device)
    sorted_cap_lens = sorted_cap_lens.to(device)
    return captions, sorted_cap_lens, sorted_cap_indices


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs


def save_img(img, path):
    im = img.data.cpu().numpy()
    # [-1, 1] --> [0, 255]
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)
    im.save(path)