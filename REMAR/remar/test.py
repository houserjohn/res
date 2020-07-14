import os
import sys
import time
import datetime
import json

import numpy as np
import torch
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, \
    ExponentialLR

from remar.utils import get_args, load_embeddings, amazon_reader, initialize_model_, \
                        prepare_minibatch, get_minibatch, print_parameters, pad, evaluate_loss

from remar.vocabulary import Vocabulary
from remar.vocabulary import UNK_TOKEN, PAD_TOKEN
from remar.models.model_helpers import build_model
from remar.models.multi_aspects import MultiAspectsLatentRationaleModel
from torch.utils.tensorboard import SummaryWriter

def test():
    """
    Main testing loop.
    """
    

    cfg = get_args()
    cfg = vars(cfg)

    device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    for k, v in cfg.items():
        print("{:20} : {:10}".format(k, str(v)))

    num_iterations = cfg["num_iterations"]
    print_every = cfg["print_every"]
    eval_every = cfg["eval_every"]
    batch_size = cfg["batch_size"]
    eval_batch_size = cfg.get("eval_batch_size", batch_size)
    aspects = cfg["aspect"]
    
    # Load the data into memory.
    print("Loading data")
   
    all_data = list(amazon_reader(
        cfg["train_path"], max_len=0))
    lengths = np.array([len(ex.tokens) for ex in all_data])
    maxlen = lengths.max()

    #train test split
    test_data = all_data[:len(all_data)// 5]
    print("test", len(test_data))
    train_data = all_data[len(test_data):]
    print("train", len(train_data))

    iters_per_epoch = len(train_data) // batch_size

    if eval_every == -1:
        eval_every = iters_per_epoch
        print("eval_every set to 1 epoch = %d iters" % eval_every)

    if num_iterations < 0:
        num_iterations = -num_iterations * iters_per_epoch
        print("num_iterations set to %d iters" % num_iterations)


    print("Loading pre-trained word embeddings")
    vocab = Vocabulary()
    vectors = load_embeddings(cfg["embeddings"], vocab)
    pad_idx = vocab.w2i[PAD_TOKEN]

    # build model
    print("Building model")    
    model = MultiAspectsLatentRationaleModel(aspects, cfg, vocab)
    initialize_model_(model)
    # # load pre-trained word embeddings
    # with torch.no_grad():
    #     model.embed.weight.data.copy_(torch.from_numpy(vectors))
    #     print("Embeddings fixed: {}".format(cfg["fix_emb"]))
    #     model.embed.weight.requires_grad = not cfg["fix_emb"]

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    scheduler = ExponentialLR(optimizer, gamma=cfg["lr_decay"])
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    #print model and parameters
    print("---------------------------------------------")
    print(model)
    print("---------------------------------------------")
    print_parameters(model)

    writer = SummaryWriter(log_dir=cfg["save_path"])  # TensorBoard
    start = time.time()
    iter_i = 0
    epoch = 0
    best_eval = 0.
    best_iter = 0
    pad_idx = vocab.w2i[PAD_TOKEN]

    # evaluate on test with best model
    print("Loading best model")
    path = os.path.join(cfg["save_path"], "Automotive-"  + str(aspects) + "-model.pt")
    ckpt = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt["state_dict"])

    print("Evaluating..", str(datetime.datetime.now()))
    model.eval()
    test_eval = evaluate_loss(
                    model, test_data, batch_size=eval_batch_size,
                    device=device, cfg=cfg)



if __name__ == "__main__":
    test()
