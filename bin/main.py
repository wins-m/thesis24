"""
Run the whole project and report tables & graphs.

Steps:
------

I. sentence preparation: (period, fundcode)
    `data_process`
II. model embeddings: (period, stockcode)
    `stock_to_vec`
III. similarity based on embeddings; other baselines
    `similarity`
    `similarity_linear`
IV. evaluation: return, volatility
    `evaluation`
IV. data description for 1~3

"""
from data_process import process_data
from stock_to_vec import stock_to_vec
from similarity import similarity, similarity_linear


def evaluation():
    """
    评估嵌入表示的有效性.
    1. 聚类的有效性: 关于行业/风格的可视化
    2. 对收益率 (协方差) 相似度预测的有效性; 和线性模型的对比
    3. 对预测公募后续持仓的有效性

    输出: 表格, 图像
    """
    pass


def main():
    K = 10

    # I. sentence preparation: (period, fundcode)
    d_sentence, d_holdamt = process_data(K=K, force_update=0)
    print(d_sentence, d_holdamt)

    # II. model embeddings: (period, stockcode)
    d_stock_vector = stock_to_vec(src=d_sentence, stime=20130101, etime=20221231,
                                  size=30, window=5, min_count=K,
                                  workers=4, skipgram=True,
                                  force_update=1)
    print(d_stock_vector)
    
    # III. similarity based on embeddings; other baselines
    d_similarity = similarity(src=d_stock_vector)

    # IV. evaluation: return, volatility

    # IV. data description for 1~3


if __name__ == '__main__':
    main()
