import pandas as pd
import tqdm
import dill
import tensorflow_hub as hub
import tensorflow_text
import tensorflow as tf #tensorflow 2.1.0

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

_model = hub.load('https://tfhub.dev/google/universal-sentence-encoder-multilingual/3')

bs = 50000
with open('data/thwiki_titles.pkl','rb') as f: thwiki_titles = dill.load(f)
with open('data/enwiki_titles.pkl','rb') as f: enwiki_titles = dill.load(f)
    
dfs = []
for i in tqdm.tqdm(range(len(thwiki_titles)//bs+1)):
    tmps = []
    for j in tqdm.tqdm(range(len(enwiki_titles)//bs+1)):
        matched_sentences, scores = match_sentences(thwiki_titles[i*bs:(i+1)*bs],\
                                                       enwiki_titles[j*bs:(j+1)*bs], _model)
        df = pd.DataFrame({'en_text':matched_sentences,'th_text':thwiki_titles[i*bs:(i+1)*bs],'use_score':scores})
        tmps.append(df)
    tmp_df = pd.concat(tmps).dropna().drop_duplicates().reset_index(drop=True)
    tmp_df['rnk'] = tmp_df.sort_values('use_score',ascending=False).groupby('th_text').cumcount()+1
    tmp_df = tmp_df[tmp_df.rnk==1]
    dfs.append(tmp_df)
    
final_df = pd.concat(dfs).dropna().drop_duplicates().reset_index(drop=True)
final_df['rnk'] = final_df.sort_values('use_score',ascending=False).groupby('th_text').cumcount()+1
final_df = final_df[final_df.rnk==1]
final_df.to_csv('data/final.csv',index=False)
