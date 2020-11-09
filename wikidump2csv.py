import argparse
import glob
import json
import pandas as pd
from tqdm.auto import tqdm
from preprocess import process_clean

# from pythainlp.tokenize import sent_tokenize as sent_tokenize_th
# from nltk.tokenize import sent_tokenize as sent_tokenize_en

# #debug
# class A:
#     def __init__(self):
#         self.input_dir = "data/thwiki/wiki_extr/th/*/*"
#         self.output_path = "data/thwiki.csv"
# args = A()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    fnames = glob.glob(args.input_dir)
    js = []
    for fname in tqdm(fnames):
        with open(fname, "r") as f:
            lines = f.readlines()
            for l in lines:
                j = json.loads(l)
                j["text"] = process_clean(j["text"])
                js.append(j)
    df = pd.DataFrame(js).dropna()
    df.to_csv(args.output_path, index=False)
