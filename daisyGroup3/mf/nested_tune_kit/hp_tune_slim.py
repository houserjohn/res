import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from hyperopt import hp, tpe, fmin

from daisy.model.SLiMRecommender import SLIM
from daisy.utils.loader import load_rate, split_test, split_validation, get_ur
from daisy.utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, mrr_at_k, ndcg_at_k

def sigmoid(x):
    return 1/(1 + np.exp(-x))

parser = argparse.ArgumentParser(description='SLiM with Coordinate Descend recommender test')
# tune settings
parser.add_argument('--sc_met', 
                    type=str, 
                    default='ndcg', 
                    help='use which metric to define hyperopt score')
# common settings
parser.add_argument('--dataset', 
                    type=str, 
                    default='ml-100k', 
                    help='select dataset')
parser.add_argument('--prepro', 
                    type=str, 
                    default='origin', 
                    help='dataset preprocess op.: origin/5core/10core')
parser.add_argument('--topk', 
                    type=int, 
                    default=10, 
                    help='top number of recommend list')
parser.add_argument('--test_method', 
                    type=str, 
                    default='tfo', 
                    help='method for split test,options: loo/fo/tfo/tloo')
parser.add_argument('--test_size', 
                    type=float, 
                    default=.2, 
                    help='split ratio for test set')
parser.add_argument('--val_method', 
                    type=str, 
                    default='tfo', 
                    help='validation method, options: cv, tfo, loo, tloo')
parser.add_argument('--fold_num', 
                    type=int, 
                    default=5, 
                    help='No. of folds for cross-validation')
parser.add_argument('--cand_num', 
                    type=int, 
                    default=1000, 
                    help='No. of candidates item for predict')
# algo settings
parser.add_argument('--alpha', 
                    type=float, 
                    default=0.02, 
                    help='elastic net parameter')
parser.add_argument('--elastic', 
                    type=float, 
                    default=0.5, 
                    help='ratio if lasso result, 0 for ridge-regression, 1 for lasso-regression')
args = parser.parse_args()

tune_log_path = './tune_log/'
if not os.path.exists(tune_log_path):
    os.makedirs(tune_log_path)

f = open(tune_log_path + f'slim_{args.dataset}_{args.prepro}_{args.val_method}.csv', 
         'w', 
         encoding='utf-8')
f.write('Pre,Rec,HR,MAP,MRR,NDCG,elastic,alpha' + '\n')

'''Validation Process for Parameter Tuning'''
# df, user_num, item_num = load_rate(args.dataset, args.prepro, binary=False)
# train_set, test_set = split_test(df, args.test_method, args.test_size)

# temporary used for tuning test result
train_set = pd.read_csv(f'./experiment_data/train_{args.dataset}_{args.prepro}_{args.test_method}.dat')
test_set = pd.read_csv(f'./experiment_data/test_{args.dataset}_{args.prepro}_{args.test_method}.dat')
if args.dataset in ['yelp']:
    train_set['timestamp'] = pd.to_datetime(train_set['timestamp'])
    test_set['timestamp'] = pd.to_datetime(test_set['timestamp'])

df = pd.concat([train_set, test_set], ignore_index=True)
user_num = df['user'].nunique()
item_num = df['item'].nunique()

# get ground truth
test_ur = get_ur(test_set)
total_train_ur = get_ur(train_set)

train_set_list, val_set_list, fn = split_validation(train_set, 
                                                    args.val_method, 
                                                    args.fold_num)

# initial candidate item pool
item_pool = set(range(item_num))
candidates_num = args.cand_num

space = {
    'elastic': hp.uniform('elastic', 0, 1),
    'alpha': hp.loguniform('alpha', np.log(1e-4), np.log(0.01)),
}

metric_idx = {
    'precision': 0,
    'recall': 1,
    'hr': 2,
    'map': 3,
    'mrr': 4, 
    'ndcg': 5,
}

def opt_func(params, mi=args.sc_met, topk=args.topk):
    elastic, alpha = params['elastic'], params['alpha']
    print(f'Parameters: elastic:{elastic} alpha:{alpha}')

    # store metrics result for test set
    fnl_metric = []
    for fold in range(fn):
        print(f'Start Validation [{fold + 1}]......')
        train = train_set_list[fold]
        validation = val_set_list[fold]

        # get ground truth
        train_ur = get_ur(train)
        val_ur = get_ur(validation)

        # build recommender model
        # model = SLIM(user_num, item_num, alpha=alpha, lam_bda=elastic, 
        #             max_iter=args.epochs, tol=args.tol)
        model = SLIM(user_num, item_num, l1_ratio=elastic, alpha=alpha)
        model.fit(train)

        # build candidates set
        val_ucands = defaultdict(list)
        for k, v in val_ur.items():
            sample_num = candidates_num - len(v) if len(v) < candidates_num else 0
            sub_item_pool = item_pool - v - train_ur[k] # remove GT & interacted
            sample_num = min(len(sub_item_pool), sample_num)
            samples = random.sample(sub_item_pool, sample_num)
            val_ucands[k] = list(v | set(samples))

        # get predict result
        preds = {}
        for u in tqdm(val_ucands.keys()):
            pred_rates = [model.predict(u, i) for i in val_ucands[u]]
            rec_idx = np.argsort(pred_rates)[::-1][:topk]
            top_n = np.array(val_ucands[u])[rec_idx]
            preds[u] = top_n

        # convert rank list to binary-interaction
        for u in preds.keys():
            preds[u] = [1 if i in val_ur[u] else 0 for i in preds[u]]

        # calculate metrics for validation set
        pre_k = np.mean([precision_at_k(r, topk) for r in preds.values()])
        rec_k = recall_at_k(preds, val_ur, topk)
        hr_k = hr_at_k(preds, val_ur)
        map_k = map_at_k(preds.values())
        mrr_k = mrr_at_k(preds, topk)
        ndcg_k = np.mean([ndcg_at_k(r, topk) for r in preds.values()])

        tmp_metric = np.array([pre_k, rec_k, hr_k, map_k, mrr_k, ndcg_k])
        fnl_metric.append(tmp_metric)

    # get final validation metrics result by average operation
    fnl_metric = np.array(fnl_metric).mean(axis=0)
    print('='*20, 'Metrics for All Validation', '='*20)
    print(f'Precision@{topk}: {fnl_metric[0]:.4f}')
    print(f'Recall@{topk}: {fnl_metric[1]:.4f}')
    print(f'HR@{topk}: {fnl_metric[2]:.4f}')
    print(f'MAP@{topk}: {fnl_metric[3]:.4f}')
    print(f'MRR@{topk}: {fnl_metric[4]:.4f}')
    print(f'NDCG@{topk}: {fnl_metric[5]:.4f}')

    score = fnl_metric[metric_idx[mi]]

    # record all tuning result and settings
    fnl_metric = [f'{mt:.4f}' for mt in fnl_metric]
    line = ','.join(fnl_metric) + f',{elastic},{alpha}' + '\n'

    f.write(line)
    f.flush()

    return -score

if __name__ == '__main__':
    best = fmin(opt_func, space, algo=tpe.suggest, max_evals=30)

    f.close()
