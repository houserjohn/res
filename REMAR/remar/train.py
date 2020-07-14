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

def train():
    """
    Main training loop.
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

    # resume from a checkpoint
    if cfg.get("ckpt", ""):
        print("Resuming from ckpt: {}".format(cfg["ckpt"]))
        ckpt = torch.load(cfg["ckpt"])
        model.load_state_dict(ckpt["state_dict"])
        best_iter = ckpt["best_iter"]
        best_eval = ckpt["best_eval"]
        iter_i = ckpt["best_iter"]
        optimizer.load_state_dict(ckpt["optimizer_state_dict"]) 

    # main training loop
    while True:  # when we run out of examples, shuffle and continue
        for batch in get_minibatch(train_data, batch_size=batch_size,
                                shuffle=True):
            # forward pass
            model.train()
        
            x, targets, _, words, batch_maxlen = prepare_minibatch(batch, model.vocab, device=device)
            output, z_matrix = model(x)

            mask = (x != pad_idx)
            # assert pad_idx == 1, "pad idx"
            classification_loss, total_loss, z_loss, z_matrix_loss = model.get_loss(output, targets, z_matrix, mask=mask)
            model.zero_grad()
            
   
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=cfg["max_grad_norm"])
            optimizer.step()
            iter_i += 1

            # print(iter_i)

            # print info
            if iter_i % 1 == 0:

                # print main loss, lr, and optional stuff defined by the model
                writer.add_scalar('train/classification_loss', classification_loss.item(), iter_i)
                writer.add_scalar('train/total_loss', total_loss.item(), iter_i)
                writer.add_scalar('train/z_loss', z_loss, iter_i)
                writer.add_scalar('train/z_matrix_loss',z_matrix_loss, iter_i)
                cur_lr = scheduler.optimizer.param_groups[0]["lr"]
                writer.add_scalar('train/lr', cur_lr, iter_i)

                # print info to console
                classification_loss_str = "%.4f" % classification_loss.item()
                total_loss_str = "%.4f" % total_loss.item()
                z_loss_str = "%.4f" % z_loss
                z_matrix_loss_str = "%.4f" % z_matrix_loss
                output = output * 6
                output = torch.round(output)
                output = torch.clamp(output, min=1, max=5)
                correct = (output == targets).sum().item()
                total_examples = targets.size(0)
                accuracy = correct / float(total_examples)
                writer.add_scalar('train/accuracy', accuracy, iter_i)
                # opt_str = make_kv_string(loss_optional)
                seconds_since_start = time.time() - start
                hours = seconds_since_start / 60 // 60
                minutes = seconds_since_start % 3600 // 60
                seconds = seconds_since_start % 60

                print("Epoch %03d Iter %08d time %02d:%02d:%02d classification loss %s z_loss %s total loss %s z_matrix loss %s accuracy %f lr %f" %
                      (epoch, iter_i, hours, minutes, seconds,
                       classification_loss_str, z_loss_str, total_loss_str,z_matrix_loss_str, accuracy, cur_lr))

            # take epoch step (if using MultiStepLR scheduler)
            if iter_i % iters_per_epoch == 0:

                cur_lr = scheduler.optimizer.param_groups[0]["lr"]
                if cur_lr > cfg["min_lr"]:
                    # if isinstance(scheduler, MultiStepLR):
                    #     scheduler.step()
                    # elif isinstance(scheduler, ExponentialLR):
                    scheduler.step()

                cur_lr = scheduler.optimizer.param_groups[0]["lr"]
                print("#lr", cur_lr)
                scheduler.optimizer.param_groups[0]["lr"] = max(cfg["min_lr"],
                                                                cur_lr)


            # evaluate
            if iter_i % eval_every == 0:
                # print("Evaluation starts - %s" % str(datetime.datetime.now()))
                # words_matrix = [pad(w, z_matrix.shape[1], "") for w in words] * aspects
                # for ii in range(z_matrix.shape[0]):
                #     for jj in range(z_matrix.shape[1]):
                #         if z_matrix[ii][jj] != 0. :
                #             print(words_matrix[ii][jj])
                #     print("-----")    
                # writer.add_text('train/rationale_example', words_matrix, iter_i)

                print("Evaluating..", str(datetime.datetime.now()))
                model.eval()

                test_eval = evaluate_loss(
                    model, test_data, batch_size=eval_batch_size,
                    device=device, cfg=cfg)

                for k, v in test_eval.items():
                    writer.add_scalar('test/' + k, v, iter_i)
                
                 # save best model parameters (lower is better)
                # compare_obj = test_eval["classification_loss"]
                compare_obj = test_eval["accuracy"]
                dynamic_threshold = best_eval * (1 - cfg["threshold"])
                accuracy = test_eval["accuracy"]
                print(accuracy)
                # only update after first 5 epochs (for stability)
                if compare_obj > dynamic_threshold \
                        and iter_i > 5 * iters_per_epoch:
                    print("new highscore", compare_obj)
                    best_eval = compare_obj
                    best_iter = iter_i
                    print("Save model")
                    if not os.path.exists(cfg["save_path"]):
                        os.makedirs(cfg["save_path"])

                    for k, v in test_eval.items():
                        writer.add_scalar('best/test/' + k, v, iter_i)

                    ckpt = {
                        "state_dict": model.state_dict(),
                        "cfg": cfg,
                        "best_eval": best_eval,
                        "best_iter": best_iter,
                        "optimizer_state_dict": optimizer.state_dict()
                    }

                    path = os.path.join(cfg["save_path"], "Music-"  + str(aspects) + "-model.pt")
                    torch.save(ckpt, path)
                
                # update lr scheduler
                if isinstance(scheduler, ReduceLROnPlateau):
                    if iter_i > 5 * iters_per_epoch:
                        scheduler.step(compare_obj)

    
            # done training
            cur_lr = scheduler.optimizer.param_groups[0]["lr"]
 
            if iter_i == num_iterations:
                print("Done training")
                print("Last lr: ", cur_lr)
            
                # evaluate on test with best model
                print("Loading best model")
                path = os.path.join(cfg["save_path"], "model.pt")
                ckpt = torch.load(path)
                model.load_state_dict(ckpt["state_dict"])

                print("Evaluating")
                test_eval = evaluate_loss(
                    model, test_data, batch_size=eval_batch_size,
                    device=device, cfg=cfg)

                test_precision = 0.
                test_macro_prec = 0.
                test_eval["precision"] = test_precision
                test_eval["macro_precision"] = test_macro_prec

                test_s = make_kv_string(test_eval)

                print("best model iter {:d} test {}".format(
                    best_iter, test_s))

                # save result
                result_path = os.path.join(cfg["save_path"], "results.json")

                cfg["best_iter"] = best_iter

                for name, eval_result in zip("test", test_eval):
                    for k, v in eval_result.items():
                        cfg[name + '_' + k] = v

                with open(result_path, mode="w") as f:
                    json.dump(cfg, f)

                # close Summary Writer
                writer.close()
                return

        epoch += 1

if __name__ == "__main__":
    train()
