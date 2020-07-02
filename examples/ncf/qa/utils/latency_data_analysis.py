import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.3f}'.format

filenames = glob.glob('inference*.json')
filenames = sorted(filenames)

warmup = 20
def latency_table(data, qs, warmup=20):
    latencies = sorted(data['run']['inference_latencies'][warmup:])
    result = OrderedDict()
    result['average_latency'] = np.mean(latencies)
    for q in qs:
        result['latency_{}_confidence'.format(q)] = np.percentile(latencies, q)
    return result

bins = np.arange(0.3, 0.6, 0.001)
fig, axes = plt.subplots(len(filenames), 1, figsize=(15, 3 * len(filenames)), sharex=True)
table_data = []

for i, f in enumerate(filenames):
    ax = axes[i]
    data = json.load(open(f))

    t = latency_table(data, qs=[50, 90, 95, 99, 100], warmup=warmup)
    t['precision'] = 'FP32' if data['run']['args_opt_level'] == 'O0' else 'FP16'
    t['batch_size'] = data['run']['args_batch_size']
    table_data.append(t)

    latencies_ms = np.array(data['run']['inference_latencies'][warmup:]) * 1000
    _ = ax.hist(latencies_ms, bins=bins, label=os.path.basename(f))
    ax.legend()
    ax.grid()
    ax.set_ylabel('# experiments')

plt.xlabel('latency [ms]')
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('latencies.pdf')

df = pd.DataFrame(table_data)
for c in df.columns:
    if 'latency' in c:
        df[c] = df[c] * 1000

columns = ["batch_size", "precision", "average_latency", "latency_50_confidence",
           "latency_90_confidence", "latency_95_confidence",
           "latency_99_confidence", "latency_100_confidence"]
df = df[columns]
df = df.sort_values(by=['batch_size', 'precision'], ascending=True)
df = df.reset_index(drop=True)
print(df.to_string(index=False))