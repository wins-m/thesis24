"""
Run the whole project and report tables & graphs.

Steps:
------

I. sentence preparation: (period, fundcode)
    `data_process`
II. model embeddings: (period, stockcode) CRUCIAL!
    `stock_to_vec`
III. similarity based on embeddings; other baselines
IV. evaluation: return, volatility
IV. data description for 1~3

"""
from data_process import process_data
from stock_to_vec import stock_to_vec


if __name__ == '__main__':
    d_sentences = process_data(K=10, force_update=True)
    d_stock_vector = stock_to_vec(src=d_sentences, stime=20130101, etime=20221231,
                                  size=30, window=5, min_count=10, workers=4, skipgram=True,
                                  force_update=True)
