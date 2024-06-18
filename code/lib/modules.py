import os, sys
import os.path as osp
import time
import random
import datetime
import argparse
from scipy.linalg import sqrtm
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
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
from lib.utils import truncated_noise
from lib.utils import mkdir_p, get_rank

from lib.datasets import TextImgDataset as Dataset
from lib.datasets import prepare_data, encode_tokens
from models.inception import InceptionV3

from torch.nn.functional import adaptive_avg_pool2d
import torch.distributed as dist

def eval(dataloader, text_encoder, netG, device, m1, s1, save_imgs, save_dir,
                times, z_dim, batch_size, truncation=True, trunc_rate=0.86):
    """ Calculates the FID """
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    model.to(device)
    model.eval()
    netG.eval()
    norm = transforms.Compose([
        transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        transforms.Resize((299, 299)),
        ])
    n_gpu = 1
    dl_length = dataloader.__len__()

    imgs_num = dl_length * n_gpu * batch_size * times
    pred_arr = np.empty((imgs_num, dims))
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop = tqdm(total=int(dl_length*times))
    for time in range(times):
        for i, data in enumerate(dataloader):
            start = i * batch_size * n_gpu + time * dl_length * n_gpu * batch_size
            end = start + batch_size * n_gpu
            imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)
            sent_emb = sent_emb.to(device)
            batch_size = sent_emb.size(0)
            netG.eval()
            with torch.no_grad():
                if truncation==True:
                    noise = truncated_noise(batch_size, z_dim, trunc_rate)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, z_dim).to(device)
                fake_imgs = netG(noise,sent_emb)
                if save_imgs==True:
                    save_single_imgs(fake_imgs, save_dir, time, dl_length, i, batch_size, keys)
                fake = norm(fake_imgs)
                pred = model(fake)[0]
                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                output = [torch.empty_like(pred) for _ in range(n_gpu)]
                for i in range(n_gpu):
                    if i == 0:
                        output[i] = pred.clone()
                    else:
                        dist.recv(output[i], src=i)
                pred_all = torch.cat(output, dim=0).squeeze(-1).squeeze(-1)
                pred_arr[start:end] = pred_all.cpu().data.numpy()
            if (n_gpu!=1) and (get_rank() != 0):
                None
            else:
                loop.update(1)
                loop.set_description(f'Evaluating:')
                loop.set_postfix()
    if (n_gpu!=1) and (get_rank() != 0):
        None
    else:
        loop.close()
    m2 = np.mean(pred_arr, axis=0)
    s2 = np.cov(pred_arr, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value


def save_single_imgs(imgs, save_dir, time, dl_len, batch_n, batch_size, keys):
    for j in range(batch_size):
        s_tmp = '%s/single/%s' % (save_dir, keys[j])
        folder = s_tmp[:s_tmp.rfind('/')]
        if not os.path.isdir(folder):
            print('Make a new folder: ', folder)
            mkdir_p(folder)
        im = imgs[j].data.cpu().numpy()
        # [-1, 1] --> [0, 255]
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        fullpath = '%s_%3d.png' % (s_tmp, time)
        im.save(fullpath)


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    diff = mu1 - mu2
    # Product might be almost singular
    covmean = sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    fid = (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fid


def sample_one_batch(noise, sent, netG, multi_gpus, epoch, img_save_dir, writer):
    fixed_results = generate_samples(noise, sent, netG)
    if (multi_gpus==True) and (get_rank() != 0):
        None
    else:
        if writer!=None:
            fixed_grid = make_grid(fixed_results.cpu(), nrow=8, range=(-1, 1), normalize=True)
            writer.add_image('fixed results', fixed_grid, epoch)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = osp.join(img_save_dir, img_name)
        vutils.save_image(fixed_results.data, img_save_path, nrow=8, range=(-1, 1), normalize=True)


def generate_samples(noise, caption, model):
    with torch.no_grad():
        fake = model(noise, caption)
    return fake
