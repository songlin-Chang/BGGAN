import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from lib.utils import mkdir_p, get_rank, load_model_weights
from models.DAMSM import RNN_ENCODER, CNN_ENCODER
from models.GAN import NetGTI, NetGIT, NetD


def prepare_models(args):
    device = args.device
    n_words = args.vocab_size
    # image encoder
    image_encoder = CNN_ENCODER(args.TEXT.EMBEDDING_DIM)
    img_encoder_path = args.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location='cpu')
    image_encoder = load_model_weights(image_encoder, state_dict, multi_gpus=False)
    # image_encoder.load_state_dict(state_dict)
    image_encoder.to(device)
    for p in image_encoder.parameters():
        p.requires_grad = False
    image_encoder.eval()
    # text encoder
    text_encoder = RNN_ENCODER(n_words, nhidden=args.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(args.TEXT.DAMSM_NAME, map_location='cpu')
    text_encoder = load_model_weights(text_encoder, state_dict, multi_gpus=False)
    text_encoder.cuda()
    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()

    netGTI = NetGTI(args.nf, args.z_dim, args.cond_dim, args.imsize, args.ch_size).to(device)
    netGIT = NetGIT(args.nf, args.imsize, args.ch_size).to(device)
    netD = NetD(args.nf, args.cond_dim).to(device)
    return image_encoder, text_encoder, netGTI, netGIT, netD


def prepare_dataset(args, split, transform):
    imsize = args.imsize #256
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(int(imsize * 76 / 64)),
            transforms.RandomCrop(imsize),
            transforms.RandomHorizontalFlip()])
    # train dataset
    from lib.datasets import TextImgDataset as Dataset
    dataset = Dataset(split=split, transform=image_transform, args=args)
    return dataset


def prepare_datasets(args, transform):
    train_dataset = prepare_dataset(args, split='train', transform=transform)
    val_dataset = prepare_dataset(args, split='val', transform=transform)
    return train_dataset, val_dataset


def prepare_dataloaders(args, transform=None):
    batch_size = args.batch_size
    num_workers = args.num_workers
    train_dataset, valid_dataset = prepare_datasets(args, transform)


    train_sampler = None
    train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, drop_last=True,
    num_workers=num_workers, shuffle='True')


    valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, drop_last=True,
    num_workers=num_workers, shuffle='True')
    return train_dataloader, valid_dataloader, \
            train_dataset, valid_dataset, train_sampler

