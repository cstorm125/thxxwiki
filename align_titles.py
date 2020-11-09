import argparse
import pandas as pd
from tqdm.auto import tqdm
import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf  # tensorflow 2.1.0

# #debug
# class A:
#     def __init__(self):
#         self.max_n=3
#         self.use_thres=0.7
#         self.max_size=max_size
#         self.en_dir = 'raw_data/economic_outlook/en_data2/'
#         self.th_dir = 'raw_data/economic_outlook/th_data2/'
#         self.output_path = 'cleaned_data/pdf_sentences.csv'
# args = A()


def match_sentences(lang1_sentences, lang2_sentences, model):
    embedding_1 = model(lang1_sentences)
    embedding_2 = model(lang2_sentences)
    distance_matrix_12 = tf.matmul(embedding_1, embedding_2, transpose_b=True)
#     print(embedding_1.shape, embedding_2.shape, distance_matrix_12.shape)
    best_distances = tf.argmax(distance_matrix_12, axis=1).numpy()

    matched_sentences_lang2 = []
    scores = []
    for i, lang2_idx in enumerate(best_distances):
        score = distance_matrix_12[i][lang2_idx].numpy()
        scores.append(score)
        matched_sentences_lang2.append(lang2_sentences[lang2_idx])
    return matched_sentences_lang2, scores


_model = hub.load(
    "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_titles_path", type=str)
    parser.add_argument("--th_titles_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--bs", type=int, default=3000)
    args = parser.parse_args()

    thwiki_titles = list(pd.read_csv(args.th_titles_path)["title"])
    enwiki_titles = list(pd.read_csv(args.en_titles_path)["title"])

    dfs = []
    for i in tqdm(range(len(thwiki_titles) // args.bs + 1)):
        tmps = []
        for j in tqdm(range(len(enwiki_titles) // args.bs + 1)):
            matched_sentences, scores = match_sentences(
                thwiki_titles[i * args.bs : (i + 1) * args.bs],
                enwiki_titles[j * args.bs : (j + 1) * args.bs],
                _model,
            )
            df = pd.DataFrame(
                {
                    "en_text": matched_sentences,
                    "th_text": thwiki_titles[i * args.bs : (i + 1) * args.bs],
                    "use_score": scores,
                }
            )
            tmps.append(df)
        tmp_df = (
            pd.concat(tmps).dropna().drop_duplicates().reset_index(drop=True)
        )
        tmp_df["rnk"] = (
            tmp_df.sort_values("use_score", ascending=False)
            .groupby("th_text")
            .cumcount()
            + 1
        )
        tmp_df = tmp_df[tmp_df.rnk == 1]
        dfs.append(tmp_df)

    final_df = pd.concat(dfs).dropna().drop_duplicates().reset_index(drop=True)
    final_df["rnk"] = (
        final_df.sort_values("use_score", ascending=False)
        .groupby("th_text")
        .cumcount()
        + 1
    )
    final_df = final_df[final_df.rnk == 1]
    final_df.to_csv(args.output_path, index=False)
