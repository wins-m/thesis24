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


def stock_to_vec(src: str,  # 语料文件地址
                 stime=20220701,  # 训练期开始时间
                 etime=20221231,  # 训练期结束时间
                 size=30,  # 表征向量维数
                 window=5,  # Word2Vec 模型窗口长度
                 min_count=10,  # 被表示的单词最少出现过的次数
                 workers=2,  # 进程数
                 skipgram=True,  # 是否用 skipgram
                 force_update=False):
    """Get stock vector using skip-gram."""

    # Output path
    out_vec = f"{src.rsplit('.', maxsplit=1)[0]}.{stime}.{etime}.word2vec.{window}.{min_count}.vector.{size}.pkl"
    out_sim = f"{src.rsplit('.', maxsplit=1)[0]}.{stime}.{etime}.word2vec.{window}.{min_count}.vector.{size}.similarity.pkl"
    if os.path.exists(out_vec) and os.path.exists(out_sim) and (not force_update):
        return out_vec, out_sim

    # 1. Prepare sample sentences (training)
    sentences = pd.read_pickle(src)

    # Train period for vector building 
    sentences = sentences.query(f"{stime} < tradingdate <= {etime}")

    if len(sentences) == 0:
        return None, None
    print('# Sentences', len(sentences), '# Token', len(set(sentences['sentence'].sum())))

    # 2. Build stock-to-vector models
    skip_gram = Word2Vec(
        sentences=sentences['sentence'], 
        vector_size=size, window=window, min_count=min_count, 
        workers=workers,
        sg=skipgram)

    # 3. Model outputs
    word_vectors = pd.DataFrame(skip_gram.wv.vectors, index=skip_gram.wv.index_to_key)
    word_vectors.to_pickle(out_vec)

    # 获取模型中所有的词语
    all_words = sorted(skip_gram.wv.index_to_key)
    # 创建一个空字典来存储相似度结果
    similarities = {}
    # 遍历所有词语两两计算相似度
    for i in range(len(all_words) - 1):
        for j in range(i+1, len(all_words)):
            word1 = all_words[i]
            word2 = all_words[j]
            similarity_score = skip_gram.wv.similarity(word1, word2)
            similarities[(word1, word2)] = similarity_score
    similarities = pd.Series(similarities).sort_index()
    similarities.to_pickle(out_sim)

    return out_vec, out_sim

if __name__ == '__main__':
    print(f"Stock vector saved in `{stock_to_vec(src='cache/sentences.10.pkl')}`")
