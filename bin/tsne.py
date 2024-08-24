"""W2V 向量表示的行业聚类情况"""
import os
os.chdir('/home/winston/Thesis')
import sys
sys.path.append('/home/winston/Thesis/bin')
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from plt_head import *
from utils import get_industry, get_last_period


def plot1(g: pd.DataFrame):
    plt.scatter(g[0], g[1], s=1)
    plt.show()


def plot2(g: pd.DataFrame, ind_dic: dict, gn=None, fig_path=None):
    """绘制前 gn 个行业聚类"""
    if gn is None:
        gn = 5
    markers = ['D', 'o', 's', '^', 'x', 'P', '*', '+', '>', '<']
    vc = g['CODELV1'].value_counts()
    colors = plt.cm.gray(np.linspace(0, 1, gn + 1))[:-1]
    for i, code in enumerate(sorted(vc.index[:gn])):
        code_g = g.query(f"CODELV1 == {code}")
        plt.scatter(code_g[0], code_g[1],
                    label=ind_dic[code],
                    color=colors[-1-i],
                    marker=markers[i%10],
                    s=15,
                    # alpha=0.8,
                    )
    if gn < 11:
        plt.legend(title='行业类别', frameon=False,
                   bbox_to_anchor=(1, 1))
    plt.xticks([])
    plt.yticks([])
    # 显示图形
    plt.tight_layout()
    if fig_path is None:
        plt.show()
    else:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()


def data_indus_dict() -> dict:    
    return pd.read_excel('data/ashareMarketData/中信行业代码内部序号_format.xlsx', index_col=0)['中信行业名称'].to_dict()


def indus_cluster_w2v(stime:int=None, etime:int=None, src=None, group_num=10, method='tsne', mod_kw='w2v'):
    """股票向量行业聚类"""
    period = get_last_period(etime)
    if src is None:
        src = f"cache/sentences.10.{stime}.{etime}.word2vec.5.10.vector.30.pkl"
    if not os.path.exists(src):
        print('Not exist:', src)
        return
    df = pd.read_pickle(src)
    # 报告期期末机构持仓股票的行业分类
    ind0 = get_industry(period//100*100+1, period)
    ind = ind0.loc[ind0.index[-1][0]].reindex(df.index)
    ind_dic = data_indus_dict()
    if method == 'tsne':  # TSNE 降维到 2D 平面
        tsne = TSNE(n_components=2, init='pca', metric='cosine', perplexity=30)
        df = df.replace(np.inf, np.nan).replace(-np.inf, np.nan).dropna()
        df_tsne2 = pd.DataFrame(tsne.fit_transform(df.values), index=df.index, columns=[0, 1])
        g_tsne = pd.concat([df_tsne2, ind], axis=1).dropna()
        plot2(g_tsne, ind_dic, gn=group_num, fig_path=f'results/indus_cluster_{mod_kw}_tsne2_g{group_num}.{period}.png')
    elif method == 'pca':  # PCA 降维
        pca = PCA(n_components=2)
        df_pca2 = pd.DataFrame(pca.fit_transform(df.values), index=df.index, columns=[0, 1])
        g_pca = pd.concat([df_pca2, ind], axis=1)
        plot2(g_pca, ind_dic, gn=group_num, fig_path=f'results/indus_cluster_w2v_pca2_g{group_num}.{period}.png')


def main():
    # indus_cluster_w2v(src= 'cache/bert/bert.30.20210101.20220630.pkl', mod_kw='bert30', etime=20230630)
    # indus_cluster_w2v(src= 'cache/bert/bert.256.pkl', mod_kw='bert256', etime=20230630)
    # indus_cluster_w2v(src= 'cache/bert/word_vec.all.rpt.ep9.pkl', mod_kw='bert.rpt.ep9', etime=20230630)
    indus_cluster_w2v(src= 'cache/bert/word_vec.all.epoch100.ep99.pkl', mod_kw='bert.epoch100.ep99', etime=20230630)

    for st, et in zip([((fy-1)*10000+701, fy*10000+101) for fy in range(2013, 2023)],
                      [((fy-1)*10000+1231, fy*10000+630) for fy in range(2014, 2024)]):
        indus_cluster_w2v(stime=st[0], etime=et[0], group_num=10, method='tsne')
        indus_cluster_w2v(stime=st[1], etime=et[1], group_num=10, method='tsne')


if __name__ == '__main__':
    main()
