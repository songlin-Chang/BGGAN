from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import Tensor
from torch.nn import functional as F
from torchtext.vocab import Vocab
from models.model import NetGIT
from models.model import ImageEncoder
from models.model import Transformer
from lib.train_utils import seed_everything, load_json, load_netGIT, prepare_models
from lib.test_utils import parse_arguments
from lib.gpu_cuda_helper import select_device
from PIL import Image

if __name__ == "__main__":
    args = parse_arguments()

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")

    # load confuguration file
    config = load_json(args.config_path)

    # Setting some pathes
    dataset_dir = args.dataset_dir  # mscoco hdf5 and json files
    dataset_dir = Path(dataset_dir)
    checkpoints_dir = config["pathes"]["checkpoint"]
    checkpoint_name = args.checkpoint_name

    # SEED
    SEED = config["seed"]
    seed_everything(SEED)

    # load vocab
    print("loading dataset...")
    vocab: Vocab = torch.load(str(Path(dataset_dir) / "vocab.pth"))
    pad_id = vocab.stoi["<pad>"]
    sos_id = vocab.stoi["<sos>"]
    eos_id = vocab.stoi["<eos>"]
    vocab_size = len(vocab)

    # --------------- dataloader --------------- #
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),  
        normalize  
    ])

    image_path = "/mnt/workspace/image_caption_trans/data/fake1.jpg"
    image = Image.open(image_path).convert('RGB')

    imgs = transform(image)
    imgs = imgs.unsqueeze(0)
    g = torch.Generator()
    g.manual_seed(SEED)
    max_len = config["max_len"]

    print("constructing models.\n")
    # prepare some hyperparameters
    image_enc_hyperparms = config["hyperparams"]["image_encoder"]
    h, w = image_enc_hyperparms["encode_size"], image_enc_hyperparms[
        "encode_size"]
    image_seq_len = int(image_enc_hyperparms["encode_size"]**2)

    transformer_hyperparms = config["hyperparams"]["transformer"]
    transformer_hyperparms["vocab_size"] = vocab_size
    transformer_hyperparms["pad_id"] = pad_id
    transformer_hyperparms["img_encode_size"] = image_seq_len
    transformer_hyperparms["max_len"] = max_len - 1

    # construct models
    netGIT = NetGIT(32, 256, 3)

    netGIT = load_netGIT(netGIT, r'/data/bggan/code/savemodel/state_epoch_600.pth', False, False)
    image_encoder = prepare_models()
    image_enc = ImageEncoder(**image_enc_hyperparms)
    transformer = Transformer(**transformer_hyperparms)
    # load
    load_path = str(Path(checkpoints_dir) / checkpoint_name)
    state = torch.load(load_path, map_location=torch.device("cpu"))
    image_model_state = state["models"][0]
    transformer_state = state["models"][1]
    image_enc.load_state_dict(image_model_state)
    transformer.load_state_dict(transformer_state)

    image_enc.to(device).eval()
    transformer.to(device).eval()

    
    jishu = 1
    pre_json = []
    truth_json = []
    
    k=1
    imgs = imgs.to(device)
    start = torch.full(size=(1, 1),
                        fill_value=sos_id,
                        dtype=torch.long,
                        device=device)
    with torch.no_grad():
        imgs_enc = image_enc(imgs, image_encoder, netGIT)  # [1, is, ie]
        logits, attns = transformer(imgs_enc, start)
        logits: Tensor  # [k=1, 1, vsc]
        attns: Tensor  # [ln, k=1, hn, S=1, is]
        log_prob = F.log_softmax(logits, dim=2)
        log_prob_topk, indxs_topk = log_prob.topk(k, sorted=True)
        # log_prob_topk [1, 1, k]
        # indices_topk [1, 1, k]
        current_preds = torch.cat(
            [start.expand(k, 1), indxs_topk.view(k, 1)], dim=1)

    seq_preds = []
    seq_log_probs = []
    seq_attns = []
    while current_preds.size(1) <= (
            max_len - 2) and k > 0 and current_preds.nelement():
        with torch.no_grad():
            imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
            # [k, is, ie]
            logits, attns = transformer(imgs_expand, current_preds)
               

                
            log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
                # log_prob: [k, vsc]
            log_prob = log_prob + log_prob_topk.view(k, 1)
                # top k probs in log_prob[k, vsc]
            log_prob_topk, indxs_topk = log_prob.view(-1).topk(k,
                                                                   sorted=True)
                # indxs_topk are a flat indecies, convert them to 2d indecies:
                # i.e the top k in all log_prob: get indecies: K, next_word_id
            prev_seq_k, next_word_id = np.unravel_index(
                indxs_topk.cpu(), log_prob.size())
            next_word_id = torch.as_tensor(next_word_id).to(device).view(
                k, 1)
                # prev_seq_k [k], next_word_id [k]

            current_preds = torch.cat(
                (current_preds[prev_seq_k], next_word_id), dim=1)
               
        seqs_end = (next_word_id == eos_id).view(-1)
        if torch.any(seqs_end):
            seq_preds.extend(seq.tolist()
                                for seq in current_preds[seqs_end])
            seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
                # get last layer, mean across transformer heads
            attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                # [k, S, h, w]
            seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

            k -= torch.sum(seqs_end)
            current_preds = current_preds[~seqs_end]
            log_prob_topk = log_prob_topk[~seqs_end]
                # current_attns = current_attns[~seqs_end]

        # Sort predicted captions according to seq_log_probs
    specials = [pad_id, sos_id, eos_id]
    sort_list = sorted(zip(seq_preds, seq_attns, seq_log_probs), key=lambda tup: -tup[2])
    if len(sort_list) == 1:
        seq_preds, seq_attns, seq_log_probs = zip(*sorted(zip(seq_preds, seq_attns, seq_log_probs), key=lambda tup: -tup[2]))

        text_preds = [[vocab.itos[s] for s in seq if s not in specials]
                        for seq in seq_preds]
        modified_list = []
        for i, word in enumerate(text_preds[0]):
            if word == '<' and i + 2 < len(text_preds[0]) and text_preds[0][i + 1] == 'num' and text_preds[0][i + 2] == '>':
                modified_list.append('<num>')
            elif word not in ['<', 'num', '>']:
                modified_list.append(word)

        sentence = ' '.join(modified_list)
        line = str(jishu)  + '\t' + sentence
        pre_json.append(line)
        print(pre_json)

