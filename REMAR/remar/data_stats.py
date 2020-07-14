"""
Take home
    1 - some categories can be None
"""

INPUT_DIR = "./raw_datasets/"

import json
from collections import Counter

def parse_yelp():
    """draw review from `review.json` """
    
    in_dir = INPUT_DIR + "yelp/"

    all_category = []
    valid_cat = 0
    
    with open(in_dir + "business.json", "r") as fin:
        for ind, ln in enumerate(fin):
            data = json.loads(ln)

            if data["categories"]:
                all_category += [x.strip().lower() for x in 
                                data["categories"].split(", ")]
                valid_cat += 1
    
    counter = Counter(all_category)

    print(counter)
    print(valid_cat)
    print("===============")
    print([k for k, v in counter.items() if v >= 200])

parse_yelp()