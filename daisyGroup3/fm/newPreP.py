#dependencies
import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
#necesary shrink
iid2idx, uid2idx = dict(), dict()
uniq_iids, uniq_uids = [], []

review_iids, review_uids = [], []
review_set = dict()

#getting data
df = pd.read_csv(f'./data/ml-100k/ml-100k/u.data', sep='\t', names=['user', 'item', 'rating', 'timestamp'], header =None)

#iteratig over data
for index, row in df.iterrows():
    iid, uid = row['item'], row['user']

    if iid not in iid2idx:
        new_iid = "i_"+str(len(uniq_iids))
        uniq_iids.append(iid)
        iid2idx[iid] = new_iid
    else:
        new_iid = iid2idx[iid]

    if uid not in uid2idx:
        new_uid = "u_"+str(len(uniq_uids))
        uniq_uids.append(uid)
        uid2idx[uid] = new_uid
    else:
        new_uid = uid2idx[uid]

    review_iids.append(new_iid)
    review_uids.append(new_uid)

    review_set[(new_uid, new_iid)] = {
            "user": new_uid,
            "item": new_iid,
            "rating": row["rating"],
            "timestamp": row["timestamp"]
    }

G = nx.Graph()
G.add_edges_from(zip(review_uids, review_iids))

G_kcore = nx.algorithms.core.k_core(G, k=10)

G_kcore_edges = [(x,y) if x[0]=="u" else (y,x) for x,y in G_kcore.edges()]

kcore_dataset = [review_set[tp] for tp in G_kcore_edges]

kcore_df = pd.DataFrame(kcore_dataset)

pdata, ndata = train_test_split(kcore_df, test_size=1)
pdata.to_csv('sProcessedData.csv', index = False)

#train, test = train_test_split(kcore_df, test_size=0.1)
#train.to_csv('sTrainData.csv', index = False)
#test.to_csv('sTestData.csv', index = False)


