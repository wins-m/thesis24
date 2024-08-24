"""
III. similarity based on embeddings; other baselines

"""
import os
_PATH = "/home/winston/Thesis"
os.chdir(_PATH)
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def get_bert_token_vector(mod_path=None, vocab_path=None, tgt=None):
    """从 bert_pytorch 训练结果中提取嵌入向量"""
    import torch
    from bert_pytorch.dataset import WordVocab
    if mod_path is None:
        # mod_path = './cache/bert/bert.model.all.rpt.ep9'
        # mod_path = './cache/bert/bert.model.all.epoch100.ep40'
        mod_path = './cache/bert/bert.model.all.epoch100.ep99'
    if vocab_path is None:
        vocab_path = './cache/bert/vocab.all'
    if tgt is None:
        tgt = mod_path.replace('bert.model', 'word_vec') + '.pkl'
    bert = torch.load(mod_path)
    vocab = WordVocab.load_vocab(vocab_path)
    word_vec = pd.DataFrame(
        bert.embedding.token.weight.detach().numpy(),
        index=vocab.itos
    ).iloc[5:]
    df = word_vec.apply(lambda s: s / s.pow(2).sum()**0.5, axis=1)
    df.to_pickle(tgt)
    print(tgt)


def get_bert_similarity(src=None, tgt=None):
    """根据向量计算余弦相似度"""
    if src is None:
        # src = 'cache/bert/word_vec.all.rpt.ep9.pkl'
        # src = 'cache/bert/word_vec.all.epoch100.ep40.pkl'
        src = 'cache/bert/word_vec.all.epoch100.ep99.pkl'
    if tgt is None:
        tgt = src.replace('word_vec', 'sim_mat')
    df = pd.read_pickle(src)
    sim_mat = (df @ df.T).copy()
    sim_mat.index.name, sim_mat.columns.name = 'stockcode', 'stockcode_right'
    sim_mat.to_pickle(tgt)
    print(tgt)


def similarity_manual(src=None, tgt=None):
    """
    人工刻画相似度
    
    """
    if src is None:
        src = 'cache/holdamt.10.pkl'
    print(src)
    df: pd.DataFrame = pd.read_pickle(src)
    # >>> df
    #         tradingdate    period   fundcode  stockcode       amount  shrpct
    # 0          20130717  20130630  001775.OF  300284.SZ  10393800.75    2.21
    # 1          20130717  20130630  001775.OF  000100.SZ  10780516.02    2.29
    # 2          20130717  20130630  001775.OF  600276.SH  12175446.76    2.59
    # 3          20130717  20130630  001775.OF  601888.SH  12263795.60    2.61
    # 4          20130717  20130630  001775.OF  600062.SH  12277665.40    2.61
    # ...             ...       ...        ...        ...          ...     ...
    # 2468171    20230331  20221231  970114.OF  601717.SH   1708596.00    2.54
    # 2468172    20230331  20221231  970114.OF  688206.SH   1778365.04    2.64
    # 2468173    20230331  20221231  970114.OF  603348.SH   1778913.00    2.64
    # 2468174    20230331  20221231  970114.OF  300481.SZ   1873773.00    2.78
    # 2468175    20230331  20221231  970114.OF  300679.SZ   1968400.00    2.92

    # [2468176 rows x 6 columns]

    # 期末持仓市值 H
    if 'H' not in df.columns:
        df['H'] = df['amount'] / 10000
        # 过去 20 日成交额均值 AMT
        if 'AMT' not in df.columns:
            amt2d = pd.read_hdf('data/amount.h5', key='data')['amount'].unstack()
            amt20 = amt2d.rolling(20).mean()
            res = []
            for per in df.period.unique():
                tmp = amt20.loc[:per]
                tmp = tmp.loc[tmp.index[-1]]
                tmp = tmp.reindex(df.query(f'period=={per}').stockcode)
                tmp = tmp.rename('AMT').reset_index()
                tmp['period'] = per
                res.append(tmp)
            res = pd.concat(res)
            df['AMT'] = res['AMT'].values
            # 拥挤度 I = H / AMT
            if 'I' not in df.columns:
                df['I'] = df['H'] / df['AMT']
        df.to_pickle(src)
        print(src)

    # 股票两两计算关联度 
    df = df.sort_values(['period', 'fundcode', 'stockcode'])
    for per in df.period.unique():
        print(per)
        tgt = Path(f'cache/sim_Kab.{per}.pkl')
        if tgt.exists():
            continue
        df_per = df.query(f"period == {per}")
        res = {}
        for fund in tqdm(df_per.fundcode.unique()):
            df_per_fund = df_per.query(f'fundcode == "{fund}"')
            for i in range(len(df_per_fund) - 1):
                for j in range(i + 1, len(df_per_fund)):
                    J_ab = df_per_fund.iloc[[i,j]]['I'].min()
                    a = df_per_fund.iloc[i]['stockcode']
                    b = df_per_fund.iloc[j]['stockcode']
                    if (a, b) in res:
                        res[(a, b)] += J_ab
                    else:
                        res[(a, b)] = J_ab
        res = pd.Series(res).rename('K')
        res.index.names = ['stockcode', 'stockcode_right']
        res = res.reset_index()
        res['lnK'] = res['K'].apply(np.log)
        res['lnK1'] = res['lnK'].sub(res['lnK'].min()).div(res['lnK'].max() - res['lnK'].min())
        res['period'] = per
        res.to_pickle(tgt)


def similarity_linear(src):
    """
    用传统的线性模型 (风格向量) 的空间距离表示相似性. 

    输出: 股票两两之间的空间距离, O(N^2)
    """
    pass


if __name__ == '__main__':
    # similarity_manual()
    get_bert_token_vector()
    get_bert_similarity()
