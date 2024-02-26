"""
II. model embeddings: (period, stockcode) CRUCIAL!

1. Prepare sample sentences (training)
2. Build stock-to-vector models
    - Skip-gram from Word2Vec
    - GloVe
    - AssetBERT
3. Model outputs

"""
import os
import pandas as pd
from gensim.models import Word2Vec


_PATH = "/home/winston/Thesis"
os.chdir(_PATH)



def refined_sent(df) -> pd.DataFrame:
    return df


def stock_to_vec(src: str, 
                 stime=20130101, etime=20221231,
                 size=30, window=5, min_count=10,
                 workers=2, skipgram=True,
                 force_update=False):
    """Get stock vector using skip-gram."""

    # Output path
    tgt = f"{src.rsplit('.', maxsplit=1)[0]}.vector.{stime}.{etime}.{size}.{window}.{min_count}.pkl"
    if os.path.exists(tgt) and (not force_update):
        return tgt

    # 1. Prepare sample sentences (training)
    sentences = pd.read_pickle(src)

    # Train period for vector building 
    sentences = sentences.loc[stime:etime]

    # 2. Build stock-to-vector models
    skip_gram = Word2Vec(
        sentences=sentences, 
        vector_size=size, window=window, min_count=min_count, 
        workers=workers,
        sg=skipgram)

    # 3. Model outputs
    word_vectors = pd.DataFrame(skip_gram.wv.vectors, index=skip_gram.wv.index_to_key)
    word_vectors.to_pickle(tgt)
    
    return tgt


if __name__ == '__main__':
    print(f"Stock vector saved in `{stock_to_vec(src='cache/sentences.10.pkl')}`")
