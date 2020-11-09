import argparse
import pandas as pd
from tqdm.auto import tqdm
from pythainlp.tokenize import sent_tokenize as sent_tokenize_th
from nltk.tokenize import sent_tokenize as sent_tokenize_en

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_path", type=str)
    parser.add_argument("--th_path", type=str)
    parser.add_argument("--mappings_path", type=str)
    parser.add_argument("--output_en_dir", type=str)
    parser.add_argument("--output_th_dir", type=str)
    parser.add_argument("--use_thres", type=int, default=0.85)
    args = parser.parse_args()
    
    mappings = pd.read_csv(args.mappings_path)
    mappings.columns = ['en_title','th_title','use_score','rnk']
    mappings = mappings[mappings.use_score>0.85]
    en = pd.read_csv(args.en_path)
    en.columns = ['id','url','en_title','en_text']
    th = pd.read_csv(args.th_path)
    th.columns = ['id','url','th_title','th_text']
    
    mapped = mappings[['en_title','th_title']]\
        .merge(th[['th_title','th_text']], left_on='th_title', right_on='th_title')\
        .merge(en[['en_title','en_text']], left_on='en_title',right_on='en_title').reset_index()
    print(f'Mapped to {mapped.shape[0]} articles')
    
    mapped_dict = mapped.to_dict('records')
    to_pad = len(str(len(mapped_dict)))
    for m in tqdm(mapped_dict):
        en_texts = m['en_text'].replace('section::::','').split('\n')
        th_texts = m['th_text'].replace('section::::','').split('\n')

        en_texts2 = []
        for t in en_texts: en_texts2+=sent_tokenize_en(t)
        with open(f'{args.output_en_dir}/doc_{str(m["index"]).zfill(to_pad)}.sent','w') as f:
            f.writelines(l + '\n' for l in en_texts2)

        th_texts2 = []
        for t in th_texts: th_texts2+=sent_tokenize_th(t)
        with open(f'{args.output_th_dir}/doc_{str(m["index"]).zfill(to_pad)}.sent','w') as f:
            f.writelines(l + '\n' for l in th_texts2)
