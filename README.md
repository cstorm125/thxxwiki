# `thxxwiki` - create th-xx parallel corpus from Wikipedia dumps

## Getting Started

1. Download `th` and `xx` Wikipedia dumps; replace `xx` with your language of choice that are used to train [mUSE](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3)
```
bin/bash prepare_wiki.sh data/thwiki th
bin/bash prepare_wiki.sh data/xxwiki xx
```
2. Create `thwiki.csv` and `xxwiki.csv` from Wikipedia dumps

```
python wikidump2csv.py --input_dir 'data/thwiki/wiki_extr/th/*/*' --output_path data/thwiki.csv
python wikidump2csv.py --input_dir 'data/xxwiki/wiki_extr/xx/*/*' --output_path data/xxwiki.csv
```

3. Align titles of each Wikipedia; default cosine similarity score threshold at 0.85

```
python align_titles.py --en_titles_path data/xxwiki.csv --th_titles_path data/thwiki.csv --output_path data/mappings.csv --bs 10000
```

4. Create sentences from aligned documents.

```
python create_sentences.py --en_path data/xxwiki.csv --th_path data/thwiki.csv --mappings_path data/mappings.csv --output_en_dir data/xx_sentences --output_th_dir data/th_sentences --use_thres 0.85
```

5. Align sentences within each document; default cosine similarity score threshold at 0.7

```
python align_sentences.py --en_dir data/xx_sentences --th_dir data/th_sentences --output_path data/xxth_aligned.csv --max_n 3 --bs 10000 --use_thres 0.7
```

## `data` directory structure

```
data

#wikipedia dumps
--xxwiki
--thwiki

#sentences
--xx_sentences
----doc_0000.sent
----doc_0001.sent
...
--th_sentences
----doc_0000.sent
----doc_0001.sent
...

#csvs
xxwiki.csv
thwiki.csv
mappings.csv
xxth_aligned.csv
```