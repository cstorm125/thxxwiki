import argparse
import glob
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from preprocess import rm_useless_spaces
import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf  # tensorflow 2.1.0

# #debug
# class A:
#     def __init__(self):
#         self.max_n=3
#         self.use_thres=0.7
#         self.max_size=max_size
#         self.en_dir = 'raw_data/economic_outlook/en_data2'
#         self.th_dir = 'raw_data/economic_outlook/th_data2'
#         self.output_path = 'cleaned_data/pdf_sentences.csv'
# args = A()


def stitch_sentences(sent, max_n=3):
    res = []
    for n in range(max_n + 1):
        for i in range(len(sent) - n + 1):
            r = " ".join(sent[i : (i + n)])
            r = rm_useless_spaces(r.replace("\n", " ").strip())
            res.append((i, r))
    return res[(len(sent) + 1) :]


def match_sentences(lang1_sentences, lang2_sentences, model):
    embedding_1 = model(lang1_sentences)
    embedding_2 = model(lang2_sentences)
    distance_matrix_12 = tf.matmul(embedding_1, embedding_2, transpose_b=True)
    print(embedding_1.shape, embedding_2.shape, distance_matrix_12.shape)
    best_distances = tf.argmax(distance_matrix_12, axis=1).numpy()

    matched_sentences_lang2 = []
    scores = []
    for i, lang2_idx in enumerate(best_distances):
        score = distance_matrix_12[i][lang2_idx].numpy()
        scores.append(score)
        matched_sentences_lang2.append(lang2_sentences[lang2_idx])
    return matched_sentences_lang2, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_dir", type=str)
    parser.add_argument("--th_dir", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--max_n", type=int, default=3)
    parser.add_argument("--bs", type=int, default=3000)
    parser.add_argument("--use_thres", type=int, default=0.7)
    args = parser.parse_args()

    print("loading model...")
    # _model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    _model = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    )
    print("model loaded")

    en_paths = sorted(glob.glob(f"{args.en_dir}/*.sent"))
    th_paths = sorted(glob.glob(f"{args.th_dir}/*.sent"))
    if len(en_paths) != len(th_paths):
        raise ValueError("must have equal number of documents")
    print(f"there are {len(en_paths)} parallel docs")

    res_en_ths = []
    for en_path, th_path in tqdm(zip(en_paths, th_paths)):
        print(en_path)
        print(th_path)
        with open(en_path, "r") as f:
            sent_en = f.readlines()
            tup_en = stitch_sentences(sent_en, args.max_n)
            sent_en2 = [i[1] for i in tup_en]
            id_en = [i[0] for i in tup_en]
        with open(th_path, "r") as f:
            sent_th = f.readlines()
            tup_th = stitch_sentences(sent_th, args.max_n)
            sent_th2 = [i[1] for i in tup_th]
            id_th = [i[0] for i in tup_th]

        print(
            f"""
        {en_path}
        en sentences: {len(sent_en)}
        th sentences: {len(sent_th)}
        stitched en sentences (max_n = {args.max_n}): {len(tup_en)}
        stiched th sentences (max_n = {args.max_n}): {len(tup_th)}
        """
        )

        # skip if there's only title
        if (len(sent_en) == 1) | (len(sent_th) == 1):
            print("skipping...")
            continue

        for i in tqdm(range(len(sent_en2) // args.bs + 1)):
            for j in tqdm(range(len(sent_th2) // args.bs + 1)):
                matched_sentences_th, scores = match_sentences(
                    sent_en2[i * args.bs : (i + 1) * args.bs],
                    sent_th2[j * args.bs : (j + 1) * args.bs],
                    _model,
                )
                res_en_th = pd.DataFrame(
                    {
                        "en_text": sent_en2[i * args.bs : (i + 1) * args.bs],
                        "th_text": matched_sentences_th,
                        "use_score": scores,
                        "id_en": id_en,
                    }
                )
                res_en_th = res_en_th[(res_en_th.use_score > args.use_thres)]
                res_en_th["src"] = en_path
                res_en_ths.append(res_en_th)
                print(
                    f"{res_en_th.shape[0]} sentences above {args.use_thres} threshold"
                )

    df = (
        pd.concat(res_en_ths).dropna().drop_duplicates().reset_index(drop=True)
    )
    df["rnk"] = (
        df.sort_values("use_score", ascending=False)
        .groupby(["src","id_en"])
        .cumcount()
        + 1
    )
    df = df[df.rnk == 1]
    print(f"saving {df.shape} to {args.output_path}")
    df.to_csv(args.output_path, index=False)
