from pathlib import Path
from collections import defaultdict
from statistics import mean, pstdev
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional as F
from torchtext.vocab import Vocab
from torchvision.transforms import Normalize, Compose
from torch.utils.data import DataLoader
import sys
sys.path.append('/data/bggan/code')

from predata.dataloader import HDF5Dataset, collate_padd
from models.model import NetGIT
from models.model import ImageEncoder
from models.model import Transformer
from nlg_metrics import Metrics
from lib.train_utils import seed_everything, load_json, load_netGIT, prepare_models
from lib.test_utils import parse_arguments
from lib.gpu_cuda_helper import select_device

if __name__ == "__main__":
    args = parse_arguments()

    # device
    device = select_device(args.device)
    print(f"selected device is {device}.\n")

    # load confuguration file
    config = load_json(args.config_path)

    # Setting some pathes
    dataset_dir = args.dataset_dir  
    dataset_dir = Path(dataset_dir)
    images_path = str(dataset_dir / "test_images.hdf5")
    captions_path = str(dataset_dir / "test_captions.json")
    lengthes_path = str(dataset_dir / "test_lengthes.json")
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
    norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = Compose([norm])

    test_dataset = HDF5Dataset(hdf5_path=images_path,
                               captions_path=captions_path,
                               lengthes_path=lengthes_path,
                               pad_id=pad_id,
                               transform=transform)

    g = torch.Generator()
    g.manual_seed(SEED)
    max_len = config["max_len"]
    val_iter = DataLoader(test_dataset,
                          collate_fn=collate_padd(max_len, pad_id),
                          batch_size=1,
                          pin_memory=True,
                          num_workers=4,
                          shuffle=False)
    print("loading dataset finished.")

    # --------------- Construct models --------------- #
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

    eval_data = defaultdict(
        list, {
            "hypos_text": [],
            "refs_text": [],
            "attns": [],
            "log_prob": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
            "gleu": [],
            "meteor": []
        })
    selected_data = defaultdict(
        list, {
            "hypos_text": [],
            "refs_text": [],
            "attns": [],
            "bleu1": [],
            "bleu2": [],
            "bleu3": [],
            "bleu4": [],
            "gleu": [],
            "meteor": []
        })
    nlgmetrics = Metrics()
    bleu4 = []
    pb = tqdm(val_iter, leave=False, total=len(val_iter))
    pb.unit = "step"
    jishu = 1
    pre_json = []
    truth_json = []
    
    for imgs, cptns_all, lens in pb:
        imgs: Tensor  # images [1, 3, 256, 256]
        cptns_all: Tensor  # all 5 captions [1, lm, cn=5]
        lens: Tensor  # lengthes of all captions [1, cn=5]
        k = 1
        # start: [1, 1]
        
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
            # current_preds: [k, S]

            # get last layer, mean across transformer heads
            # attns = attns[-1].mean(dim=1).view(1, 1, h, w)  # [k=1, s=1, h, w]
            # current_attns = attns.repeat_interleave(repeats=k, dim=0)
            # [k, s=1, h, w]

        seq_preds = []
        seq_log_probs = []
        seq_attns = []
        while current_preds.size(1) <= (
                max_len - 2) and k > 0 and current_preds.nelement():
            with torch.no_grad():
                imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
                # [k, is, ie]
                
                logits, attns = transformer(imgs_expand, current_preds)
                # logits: [k, S, vsc]
                # attns: # [ln, k, hn, S, is]
                # get last layer, mean across transformer heads
                # attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                # # [k, S, h, w]
                # attns = attns[:, -1].view(k, 1, h, w)  # current word

                # next word prediction
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
                # current_attns = torch.cat(
                #     (current_attns[prev_seq_k], attns[prev_seq_k]), dim=1)

            # find predicted sequences that ends
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
            text_refs = [[vocab.itos[r] for r in ref if r not in specials]
                         for ref in cptns_all.squeeze(0).permute(1, 0)]
        
        
            # calculate scores for each prediction
            scores = defaultdict(list)
            for text_pred in text_preds:
                for k, v in nlgmetrics.calculate([text_refs], [text_pred]).items():
                    scores[k].append(v)
                    
           
                
            modified_list = []
            for i, word in enumerate(text_preds[0]):
                if word == '<' and i + 2 < len(text_preds[0]) and text_preds[0][i + 1] == 'num' and text_preds[0][i + 2] == '>':
                    modified_list.append('<num>')
                elif word not in ['<', 'num', '>']:
                    modified_list.append(word)

            sentence = ' '.join(modified_list)
            line = str(jishu)  + '\t' + sentence
            pre_json.append(line)

            modified_list1 = []
            for i, word in enumerate(text_refs[0]):
                if word == '<' and i + 2 < len(text_refs[0]) and text_refs[0][i + 1] == 'num' and text_refs[0][i + 2] == '>':
                    modified_list1.append('<num>')
                elif word not in ['<', 'num', '>']:
                    modified_list1.append(word)

            sentence1 = ' '.join(modified_list1)
            line1 = str(jishu) + '\t' + sentence1
            truth_json.append(line1)

            jishu += 1
            # save all eval data
            eval_data["hypos_text"].append(text_preds)
            eval_data["refs_text"].append(text_refs)
            eval_data["attns"].append(list(seq_attns))
            eval_data["log_prob"].append(list(seq_log_probs))
            for k, v_list in scores.items():
                eval_data[k].append(v_list)

            # save data for the prediction with the highest log_prob
            selected_data["hypos_text"].append(text_preds[0])
            selected_data["refs_text"].append(text_refs)
            selected_data["attns"].append(list(seq_attns)[0])
            selected_data["bleu1"].append(scores["bleu1"][0])
            selected_data["bleu2"].append(scores["bleu2"][0])
            selected_data["bleu3"].append(scores["bleu3"][0])
            selected_data["bleu4"].append(scores["bleu4"][0])
            selected_data["gleu"].append(scores["gleu"][0])
            selected_data["meteor"].append(scores["meteor"][0])

            # tracking some data on progress bar
            # show bleu4 for the prediction with the highest log_prob

            pb.set_description(
                f'bleu4: Current: {selected_data["bleu4"][-1]:.4f}, Max: {max(selected_data["bleu4"]):.4f}, Min: {min(selected_data["bleu4"]):.4f}, Mean: {mean(selected_data["bleu4"]):.4f} \u00B1 {pstdev(selected_data["bleu4"]):.2f}'  # noqa: E501
            )

    pb.close()
    # save
    print("\nsaveing data...")
    experiment_name = checkpoint_name.split("/")[0]
    save_dir = Path(args.save_dir) / f"{experiment_name}"
    save_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data=eval_data).to_pickle(str(save_dir / "all.pickle"))
    pd.DataFrame(data=selected_data).to_pickle(
        str(save_dir / "selected.pickle"))
    pjson = os.path.join(save_dir, 'pre_json.txt')
    tjson = os.path.join(save_dir, 'truth_json.txt')
    with open(pjson, 'w', encoding='utf8') as f_p:
        for l in pre_json:
            f_p.writelines(l + '\n')


    with open(tjson, 'w', encoding='utf8') as f_t:
        for l in truth_json:
            f_t.writelines(l + '\n')
    
    

    print("Done.")
