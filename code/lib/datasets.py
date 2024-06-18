from nltk.tokenize import RegexpTokenizer
from collections import defaultdict

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import time
import numpy as np
import pandas as pd
from io import BytesIO
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from .utils import truncated_noise


def get_one_batch_data(dataloader, text_encoder, args):
    data = next(iter(dataloader))
    imgs, sent_emb, words_embs, keys = prepare_data(data, text_encoder)
    return imgs, words_embs, sent_emb


def get_fix_data(train_dl, test_dl, text_encoder, args):
    fixed_image_train, fixed_word_train, fixed_sent_train = get_one_batch_data(train_dl, text_encoder, args)
    fixed_image_test, fixed_word_test, fixed_sent_test = get_one_batch_data(test_dl, text_encoder, args)
    fixed_image = torch.cat((fixed_image_train, fixed_image_test), dim=0)
    fixed_sent = torch.cat((fixed_sent_train, fixed_sent_test), dim=0)
    #假设添加词向量
    fix_word = torch.cat((fixed_word_train, fixed_word_test), dim=0)
    if args.truncation==True:
        noise = truncated_noise(fixed_image.size(0), args.z_dim, args.trunc_rate)
        fixed_noise = torch.tensor(noise, dtype=torch.float).to(args.device)
    else:
        fixed_noise = torch.randn(fixed_image.size(0), args.z_dim).to(args.device)
    return fixed_image, fixed_sent, fixed_noise


def prepare_data(data, text_encoder):
    imgs, captions, caption_lens, keys = data
    captions, sorted_cap_lens, sorted_cap_idxs = sort_sents(captions, caption_lens)
    sent_emb, words_embs = encode_tokens(text_encoder, captions, sorted_cap_lens)
    sent_emb = rm_sort(sent_emb, sorted_cap_idxs)
    words_embs = rm_sort(words_embs, sorted_cap_idxs)
    imgs = Variable(imgs).cuda()
    return imgs, sent_emb, words_embs, keys


def sort_sents(captions, caption_lens):
    # sort data by the length in a decreasing order
    sorted_cap_lens, sorted_cap_indices = torch.sort(caption_lens, 0, True)
    #对caption进行排序
    captions = captions[sorted_cap_indices].squeeze()
    captions = Variable(captions).cuda()
    sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    return captions, sorted_cap_lens, sorted_cap_indices


def encode_tokens(text_encoder, caption, cap_lens):
    #caption的shape为batch_size x cap_lens
    # encode text
    with torch.no_grad():
        if hasattr(text_encoder, 'module'):
            hidden = text_encoder.module.init_hidden(caption.size(0))
        else:
            #caption shape为16x18
            hidden = text_encoder.init_hidden(caption.size(0))
        #进行文本编码
        words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
    return sent_emb, words_embs 


def rm_sort(caption, sorted_cap_idxs):
    non_sort_cap = torch.empty_like(caption)
    for idx, sort in enumerate(sorted_cap_idxs):
        non_sort_cap[sort] = caption[idx]
    return non_sort_cap


def get_imgs(img_path, transform=None, normalize=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if transform is not None:
        img = transform(img)
    if normalize is not None:
        img = normalize(img)
    return img

class TextImgDataset(data.Dataset):
    def __init__(self, split='train', transform=None, args=None):
        self.transform = transform
        self.word_num = args.TEXT.WORDS_NUM
        #embeddings_num为1
        self.embeddings_num = args.TEXT.CAPTIONS_PER_IMAGE
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.split=split
        split_dir = os.path.join(self.data_dir, split)

        self.filenames, self.captions, self.ixtoword, \
            self.wordtoix, self.n_words = self.load_text_data(self.data_dir, split)

        self.class_id = self.load_class_id(split_dir, len(self.filenames))
        self.number_example = len(self.filenames)


    #制作caption文件
    def load_captions(self, data_dir, filenames):
        all_captions = []
        for i in range(len(filenames)):
            cap_path = '%s/text/%s.txt' % (data_dir, filenames[i])
            with open(cap_path, "r") as f:
                captions = f.read().encode('utf-8').decode('utf8').split('\n') #一个列表一个元素是其中的一行
                cnt = 0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    cap = cap.replace("\ufffd\ufffd", " ")
                    # picks out sequences of alphanumeric characters as tokens
                    # and drops everything else
                    #分词
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    # tokens = tokenizer.tokenize(cap)

                    # print('tokens', tokens)
                    if len(tokens) == 0:
                        print('cap', cap)
                        continue

                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    all_captions.append(tokens_new)
                    cnt += 1
                    #embeddings_num 10
                    if cnt == self.embeddings_num:
                        break
                if cnt < self.embeddings_num:
                    print('ERROR: the captions for %s less than %d'
                          % (filenames[i], cnt))

        return all_captions

    def build_dictionary(self, train_captions, test_captions):
        word_counts = defaultdict(float)
        captions = train_captions + test_captions
        for sent in captions:
            for word in sent:
                word_counts[word] += 1

        vocab = [w for w in word_counts if word_counts[w] >= 0]

        ixtoword = {}
        ixtoword[0] = '<end>'
        wordtoix = {}
        wordtoix['<end>'] = 0
        ix = 1
        #自己设定每个单词的值
        for w in vocab:
            wordtoix[w] = ix
            ixtoword[ix] = w
            ix += 1

        train_captions_new = []
        for t in train_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            train_captions_new.append(rev)

        test_captions_new = []
        for t in test_captions:
            rev = []
            for w in t:
                if w in wordtoix:
                    rev.append(wordtoix[w])
            # rev.append(0)  # do not need '<end>' token
            test_captions_new.append(rev)

        return [train_captions_new, test_captions_new,
                ixtoword, wordtoix, len(ixtoword)]

    def load_text_data(self, data_dir, split):
        filepath = os.path.join(data_dir, 'captions_DAMSM.pickle')
        train_names = self.load_filenames(data_dir, 'train')
        test_names = self.load_filenames(data_dir, 'test')
        if not os.path.isfile(filepath):
            #手动获取caption
            train_captions = self.load_captions(data_dir, train_names)
            test_captions = self.load_captions(data_dir, test_names)

            train_captions, test_captions, ixtoword, wordtoix, n_words = \
                self.build_dictionary(train_captions, test_captions)
            with open(filepath, 'wb') as f:
                pickle.dump([train_captions, test_captions,
                             ixtoword, wordtoix], f, protocol=2)
                print('Save to: ', filepath)
        else:
            with open(filepath, 'rb') as f:
                x = pickle.load(f)
                train_captions, test_captions = x[0], x[1]
                ixtoword, wordtoix = x[2], x[3]
                del x
                n_words = len(ixtoword) #5450个单词与整数映射表
                print('Load from: %s (%d)' % (filepath, n_words))
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            filenames = test_names
        return filenames, captions, ixtoword, wordtoix, n_words

    def load_class_id(self, data_dir, total_num):
        if os.path.isfile(data_dir + '/class_info.pickle'):
            with open(data_dir + '/class_info.pickle', 'rb') as f:
                class_id = pickle.load(f, encoding="bytes")
        else:
            class_id = np.arange(total_num)
        return class_id

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def get_caption(self, sent_ix):
        # a list of indices for a sentence 得到一个句子的词向量表示
        sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
        if (sent_caption == 0).sum() > 0:
            print('ERROR: do not need END (0) token', sent_caption)
        #这句话中单词的数量
        num_words = len(sent_caption)
        # pad with 0s (i.e., '<end>')
        #生成一个（10,1）的二维数组
        x = np.zeros((self.word_num, 1), dtype='int64')
        x_len = num_words
        if num_words <= self.word_num:
            x[:num_words, 0] = sent_caption
        else:
            #创建索引列表ix
            ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
            np.random.shuffle(ix)
            ix = ix[:self.word_num]
            ix = np.sort(ix)
            x[:, 0] = sent_caption[ix]
            x_len = self.word_num
        return x, x_len

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        cls_id = self.class_id[index]
        data_dir = self.data_dir
        if self.dataset_name.find('xray') != -1:
            if self.split=='train':
                img_name = '%s/images/train/%s.png' % (data_dir, key)
            else:
                img_name = '%s/images/test/%s.png' % (data_dir, key)
        elif self.dataset_name.find('mimic') != -1:
            if self.split=='train':
                img_name = '%s/images/train/%s.jpg' % (data_dir, key)
            else:
                img_name = '%s/images/test/%s.jpg' % (data_dir, key)
        else:
            img_name = '%s/images/%s.jpg' % (data_dir, key)

        imgs = get_imgs(img_name, self.transform, normalize=self.norm)
        # random select a sentence
        #sent_ix为0-256的随机整数
        sent_ix = random.randint(0, self.embeddings_num)
        #随机得到一个整数 从而取选择一个句子
        new_sent_ix = index * self.embeddings_num + sent_ix
        caps, cap_len = self.get_caption(new_sent_ix)

        return imgs, caps, cap_len, key

    def __len__(self):
        return len(self.filenames)


