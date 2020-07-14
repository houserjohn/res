# REMAR
Rating prediction with Explainable Multiple Aspect Rationale

## Prerequisite

1. Use the following cmd to create a docker

    ```bash
    $ nvidia-docker run -it --rm -v /local2/zyli/REMAR:/workspace/REMAR -v /local2/zyli/irs_fn/data/raw/yelp:/workspace/REMAR/raw_datasets/yelp  nvcr.io/nvidia/pytorch:20.01-py3
    ```

2. Use the following cmds to update pip and install requirements

   ```bash
   $ pip install --update pip
   $ cd REMAR
   $ pip install -r requirements.txt
   ```

## Run preprocessing

### For the three datasets

1. Preprocess Yelp dataset

    **Note**: Please note that `--min_cat_num` is required for Yelp.

    ```bash
    $ python remar/preprocess.py --dataset yelp --min_cat_num 2 --k_core 5
    ```

    Here the statistics of the dataset:

    | k-core | #Cat | #node-ori | #edge-ori | #node-kcore | #edge-kcore | #user-kcore | #business-kcore | ratings |
    |--------|------|-----------|-----------|-------------|-------------|-------------|-----------------|---------|
    | 5      | 2    | 1191047   | 4056654   | 220201      | 2529581     | 172922      | 47279           | 1-5     |

2. Preprocess Amazon dataset(s)

    We use the 5-core dataset in following categories: TODO. Please download them from this [link](https://nijianmo.github.io/amazon/index.html). And *RENAME* the downloaded file to `[category].json`. For example, rename `Software_5.json` to `software.json`.
    
    To parse different subset of Amazon data, just specify the name of the subset in argument `--amazon_subset`. See the following example:

    ```bash
    $ python remar/preprocess.py --dataset amazon --amazon_subset software
    ```

    Here is the statistics of the dataset for reference:
    | subset   | #entities | #reviews | #users | #items |
    |----------|-----------|----------|--------|--------|
    | software | 2628      | 12804    | 1826   | 802    |


3. Preprocess Goodreads dataset

### Supporting files

There are

    TODO

## Run Program

Run the command from the root directory (Only using Amazon dataset at the moment)

```bash
python3 -m remar.train
```

Config on model and files can be added, details are in utils.py

```bash
python3 -m remar.train \
    --model latent \
    --aspect 5 \
    --train_path data/reviews.aspect0.train.txt.gz \
    --scheduler exponential \
    --save_path results/ \
    --dependent-z \
    --selection 0.13 --lasso 0.02
```