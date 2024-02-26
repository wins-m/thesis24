"""
Run the whole project and report tables & graphs.

Steps:
------

I. sentence preparation: (period, fundcode)
    `data_process`
II. model embeddings: (period, stockcode) CRUCIAL!
    `stock_to_vec`
III. similarity based on embeddings; other baselines
IV. data description for 1~3
V. evaluation: return, volatility

"""
from data_process import process_data


if __name__ == '__main__':
    dsentences = process_data()
