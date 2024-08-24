"""
https://github.com/hobinkwak/Stock2Vec-Inverse-Volatility/blob/main/implementation.ipynb

"""
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta

from embedding import Stock2Vec
from utils import get_universe, get_price_df, get_rebal_dates, compute_portfolio_cum_rtn


def double_inverse_volatility_optimize(sub_rtn_df, reb_dt, lb=None, ub=None):
    ipv = []
    weights = []
    for i in reb_dt:
        w = ((1 / sub_rtn_df[i].std()) / (1 / sub_rtn_df[i].std()).sum()).values
        weights.append(w)
        port_vol = np.dot(np.dot(w, sub_rtn_df[i].cov()), w.T)
        inverse_port_vol = 1 / port_vol
        ipv.append(inverse_port_vol)
    ipv = np.array(ipv)
    ipv /= ipv.sum()
    final_weights = np.hstack([weights[i] * ipv[i] for i in range(len(reb_dt))])
    return final_weights


def weights_by_stock2vec(rtn_df, rebal_dates, optimize_func, 
                         days=180, wv_n_cluster=5, wv_size=100, wv_window=10,
                         wv_min_count=0,
                         wv_skipgram=True, lb=0.1, ub=0.3):
    weights = pd.DataFrame()
    for i in tqdm(range(len(rebal_dates))):
        sub_rtn_df = rtn_df.loc[rebal_dates[i] - timedelta(days=days): rebal_dates[i] - timedelta(days=1)]
        sv = Stock2Vec(sub_rtn_df.copy(), wv_n_cluster)
        df = sv.make_rtn_data()
        df = sv.sort_by_rtn(df)
        sv.train_n_save_word2vec(df, size=wv_size, window=wv_window, min_count=wv_min_count, skipgram=wv_skipgram)
        vectors = sv.get_sg_vectors()
        clusters = sv.kmeans_clustering(vectors)
        result = sv.extract_ticker(clusters)
        final_weights = optimize_func(sub_rtn_df, result, lb=lb, ub=ub)
        weights = pd.concat([weights, pd.DataFrame(final_weights.reshape(1, rtn_df.shape[-1]), index=[rebal_dates[i]],
                                                   columns=rtn_df.columns)])
    return weights



if __name__ == '__main__':
    univ0: pd.DataFrame = get_universe(bd='2017-01-01', ed='2023-06-30')
    price_df: pd.DataFrame = get_price_df(univ=univ0)  # (date, instrument, adjclose)
    rebal_dates: pd.DataFrame = get_rebal_dates(price_df)
    rtn_df: pd.DataFrame = price_df.pct_change(fill_method=None).iloc[1:]  # (date, instrument, rtn_c2c)

    stock2vec_db_iv = weights_by_stock2vec(rtn_df, rebal_dates, double_inverse_volatility_optimize)
    cum_rtn2 = compute_portfolio_cum_rtn(price_df, stock2vec_db_iv).sum(axis=1)
