"""

"""
import os
os.chdir('/home/winston/Thesis')
import numpy as np
import pandas as pd
import sys
sys.path.append('/home/winston/Thesis/bin')
from pathlib import Path
from utils import get_last_period
from plt_head import *


describe_dict = {
        'count': '观测数',
        'mean': '均值',
        'std': '标准差',
        'min': '最小值',
        '25%': '下四分位数',
        '50%': '中位数',
        '75%': '上四分位数',
        'max': '最大值'
    }


def table_sim_w2v():
    src = Path('cache')
    # df = pd.read_pickle("cache/sentences.10.20190101.20200630.word2vec.5.10.vector.30.similarity.pkl")
    def read_data(src):
        _ = src.name.split('.')
        stime, etime = int(_[2]), int(_[3])
        if stime // 10000 == etime // 10000:
            return pd.DataFrame()
        per = get_last_period(etime)
        print(per)
        df = pd.read_pickle(src)
        df = df.reset_index()
        df.columns = ['stockcode', 'stockcode_right', 'sim']
        df['period'] = per
        return df
    df = pd.concat(
        [read_data(f) for f in
         sorted([file for file in src.iterdir() if 'vector.30.similarity.pkl' in file.name])],
        axis=0)
    # df = df.sort_values(['period', 'sim']).reset_index(drop=True)

    gkw = 'w2v'
    kw = 'sim'
    # df.query(f'period == {per}').sim.hist(bins=30, color='gray', grid=True)
    df.sim.hist(bins=30, color='gray', grid=True)
    plt.xlabel('W2V 相似度')
    plt.ylabel('观测数')
    # plt.xlim(0, 1)
    plt.tight_layout()
    tgt = f'results/fig_{gkw}_{kw}_histgram.png'
    # tgt = f'results/fig_{gkw}_{kw}_histgram.{per}.png'
    plt.savefig(tgt, transparent=True)
    print(tgt)
    plt.close()

    # 定义不同的 per 值
    periods = [20221231, 20201231, 20181231, 20161231, 20141231]
    # periods = [20221231, 20181231, 20141231]
    # 循环处理每个 per 值
    for idx, per in enumerate(periods):
        subset_df = df[df['period'] == per]
        color = 1 - (idx + 1) / len(periods)  # 通过灰度表示不同的 per
        subset_df[kw].hist(bins=30, color=str(color), alpha=0.5, label=f'报告期 {per}', grid=True)
    # 添加图例和标签
    plt.xlabel('W2V 相似度')
    plt.ylabel('观测数')
    plt.xlim(-0.4, 1)  # 根据实际数据调整 x 轴范围
    plt.legend(frameon=False)
    plt.tight_layout()
    # 保存图像
    tgt = f'results/fig_{gkw}_{kw}_histogram_{len(periods)}per.png'
    plt.savefig(tgt, transparent=True)
    print(tgt)
    plt.close()

    res = df.groupby('period')[kw].describe()
    res.loc['全历史'] = df[kw].describe()
    res['count'] = res['count'].astype(int)
    res = res.rename(columns=describe_dict)
    res.index.name = '报告期'
    res.iloc[:, 1:] = res.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")
    print(res)
    tgt = f'results/tab_{gkw}_{kw}_per_stat.xlsx'
    res.to_excel(tgt)
    print(tgt)

    info = pd.read_excel('data/全部A股.xlsx')
    info.columns = ['stockcode', 'name', 'stockname', 'indlv1']
    info = info.set_index('stockcode')

    mv = pd.read_hdf('data/ashareMarketData/日行情/流通市值.h5', key='data')

    per = 20221231
    mv1 = mv.loc[:per].iloc[-1].dropna().astype(float).div(1e4).apply(lambda x: f'{x:.1f}')
    for sc in [
        '600519.SH',
        '300750.SZ',
        '002230.SZ',
    ]:
        # sc = '600105.SH'
        # sc = '603380.SH'
        # sc = '300015.SZ'
        # sc = '600754.SH'
        # sc = '000001.SZ'
        # sc = '601888.SH'
        res = df.query(f'(period == {per}) and ((stockcode == "{sc}") or (stockcode_right == "{sc}"))')\
            .sort_values(kw, ascending=False)\
            .head(10).copy()
        res['stockname'] = res.stockcode.apply(lambda x: info.stockname.loc[x] if x in info.index else None)
        res['stockcode_other'] = res[['stockcode', 'stockcode_right']].apply(lambda s: ''.join(s).replace(sc, ''), axis=1)
        res['stockname_other'] = res.stockcode_other.apply(lambda x: info.stockname.loc[x] if x in info.index else None)
        res['indus_other'] = res.stockcode_other.apply(lambda x: info.indlv1.loc[x] if x in info.index else None)
        res['mv_other'] = res.stockcode_other.apply(lambda x: mv1.loc[x] if x in mv1.index else None)
        res = res.set_index('stockcode_other')[['stockname_other', kw, 'indus_other', 'mv_other']]
        res[kw] = res[kw].apply(lambda x: f'{x:.4f}')
        res.columns = ['股票简称', 'W2V 相似度', '所属行业', '流通市值（亿）']
        res.index.name = '股票代码'
        res = res.reset_index()
        res.loc[-1] = [sc, info.stockname.loc[sc], '-', info.indlv1.loc[sc], mv1.loc[sc]]
        res = res.sort_index().reset_index(drop=True)
        print(res)
        tgt = f'results/{gkw}_{kw}_{per}_{sc}.xlsx'
        res.to_excel(tgt)
        print(tgt)




def table_sim_lnK1():
    src = Path('cache')
    df = pd.concat(
        [pd.read_pickle(f) for f in
         sorted([file for file in src.iterdir() if 'sim_Kab' in file.name])],
        axis=0)
    
    df.lnK1.hist(bins=30, color='gray', grid=True)
    plt.xlabel('人工关联度')
    plt.ylabel('观测数')
    # plt.grid()
    plt.xlim(0, 1)
    plt.tight_layout()
    tgt = f'results/fig_man_lnK1_histgram.png'
    plt.savefig(tgt, transparent=True)
    print(tgt)
    plt.close()

    # 定义不同的 per 值
    periods = [20221231, 20201231, 20181231, 20161231, 20141231]
    periods = [20221231,20181231,20141231]
    # 循环处理每个 per 值
    for idx, per in enumerate(periods):
        subset_df = df[df['period'] == per]
        color = 1 - (idx + 1) / len(periods)  # 通过灰度表示不同的 per
        subset_df['lnK1'].hist(bins=30, color=str(color), alpha=0.6, label=f'报告期 {per}', grid=True)
    # 添加图例和标签
    plt.xlabel('人工关联度')
    plt.ylabel('观测数')
    plt.xlim(0, 1)  # 根据实际数据调整 x 轴范围
    plt.legend(frameon=False)
    plt.tight_layout()
    # plt.show()
    # 保存图像
    tgt = f'results/fig_man_lnK1_histogram_{len(periods)}per.png'
    plt.savefig(tgt, transparent=True)
    print(tgt)
    plt.close()

    res = df.groupby('period').lnK1.describe()
    res.loc['全历史'] = df.lnK1.describe()
    res['count'] = res['count'].astype(int)
    res = res.rename(columns=describe_dict)
    res.index.name = '报告期'
    res.iloc[:, 1:] = res.iloc[:, 1:].applymap(lambda x: f"{x:.2f}")
    print(res)
    tgt = 'results/tab_man_lnK1_per_stat.xlsx'
    res.to_excel(tgt)
    print(tgt)

    info = pd.read_excel('data/全部A股.xlsx')
    info.columns = ['stockcode', 'name', 'stockname', 'indlv1']
    info = info.set_index('stockcode')
    mv = pd.read_hdf('data/ashareMarketData/日行情/流通市值.h5', key='data')
    
    per = 20221231
    mv1 = mv.loc[:per].iloc[-1].dropna().astype(float).div(1e4).apply(lambda x: f'{x:.1f}')
    for sc in [
        '600519.SH',
        '300750.SZ',
        '002230.SZ',
        '300015.SZ',
        '600754.SH'
    ]:
        res = df.query(f'period == {per}').query(f'(stockcode == "{sc}") or (stockcode_right == "{sc}")').sort_values('lnK1', ascending=False).head(10).copy()
        res['stockname'] = res.stockcode.apply(lambda x: info.stockname.loc[x] if x in info.index else None)
        res['stockcode_other'] = res[['stockcode', 'stockcode_right']].apply(lambda s: ''.join(s).replace(sc, ''), axis=1)
        res['stockname_other'] = res.stockcode_other.apply(lambda x: info.stockname.loc[x] if x in info.index else None)
        res['indus_other'] = res.stockcode_other.apply(lambda x: info.indlv1.loc[x] if x in info.index else None)
        res['mv_other'] = res.stockcode_other.apply(lambda x: mv1.loc[x] if x in mv1.index else None)
        res = res.set_index('stockcode_other')[['stockname_other', 'lnK1', 'indus_other', 'mv_other']]
        res.columns = ['股票简称', '人工关联度', '所属行业', '流通市值（亿）']
        res.index.name = '股票代码'
        res = res.reset_index()
        res.loc[-1] = [sc, info.stockname.loc[sc], '-', info.indlv1.loc[sc], mv1.loc[sc]]
        res = res.sort_index().reset_index(drop=True)
        print(res)
        tgt = f'results/man_lnK1_{per}_{sc}.xlsx'
        res.to_excel(tgt)
        print(tgt)


def figure_sentence_len():
    src = 'cache/sentences.10.pkl'
    df = pd.read_pickle(src)
    res = pd.concat(
        [df.sentence.apply(len).describe().rename('语句长度'),
         df.sentence.apply(len).apply(np.log10).describe().rename('取 log10 对数')],
        axis=1).T
    res['count'] = res['count'].astype(int)
    res = res.rename(columns=describe_dict)
    print(res)
    
    df.sentence.apply(len).apply(np.log10).hist(bins=30, color='gray', grid=True)
    plt.xlabel('语句长度（取 log10 对数）')
    plt.ylabel('观测数')
    # plt.grid()
    plt.tight_layout()
    tgt = 'results/fig_sentence_length_histgram.png'
    plt.savefig(tgt, transparent=True)
    print(tgt)
    plt.close()


def figure_01(fig_path=None):
    """基金数与持仓股票数"""

    def cal_period_fund_stock_number(df):
        # 只考虑 2013~2022 的报告期（因为 20230630 可能未披露
        df = df.query("20130101 <= period <= 20221231")
        # 每个报告期的基金数量
        per_fund_num = df[['period', 'fundcode']].drop_duplicates().groupby('period')['fundcode'].count().rename('基金数')
        # 每个报告期被持仓的股票数量
        per_stock_num = df[['period', 'stockcode']].drop_duplicates().groupby('period')['stockcode'].count().rename('股票数')
        # 图: 基金数与持仓股票数
        res = pd.concat([per_fund_num, per_stock_num], axis=1).query('基金数 > 10')
        # 只保留半年度
        res = res.query('(period % 10000 in [1231, 630])')
        return res


    src1 = 'cache/data_fund.pkl'
    df1 = pd.read_pickle(src1)
    src = 'cache/holdamt.10.pkl'
    df = pd.read_pickle(src)
    res = pd.concat([cal_period_fund_stock_number(df1), cal_period_fund_stock_number(df)], axis=1)

    period = res.index.astype(str)
    fund_count = res.iloc[:, 0]
    fund_count_1 = res.iloc[:, 2]
    stock_count = res.iloc[:, 1]
    stock_count_1 = res.iloc[:, 3]

    # 创建画布和子图
    fig, ax1 = plt.subplots()
    # 绘制柱状图
    ax1.bar(period, fund_count, color='gray', alpha=0.4, label='全市场基金数（左轴）')
    ax1.bar(period, fund_count_1, color='gray', alpha=1, label='样本基金数（左轴）')
    ax1.set_xlabel('报告期')
    ax1.set_ylabel('基金数', color='gray')
    ax1.tick_params('y', colors='gray')
    ax1.set_ylim(0, 7000)
    ax1.set_yticks(range(0, 7001, 1000))  # 设置间隔为500
    ax1.set_xticks(period)
    ax1.set_xticklabels(period, rotation=45, ha='right', va='top')
    # 创建第二个y轴
    ax2 = ax1.twinx()
    # 绘制折线图
    ax2.plot(period, stock_count, color='k', linestyle='--', linewidth=3, label='基金持仓股票数（右轴）')
    ax2.plot(period, stock_count_1, color='k', marker='', linewidth=3, label='样本股票数（右轴）')
    ax2.set_ylim(0, 4900)
    ax2.set_yticks(range(0, 4901, 700))  # 设置间隔为500
    ax2.set_ylabel('股票数', color='k')
    ax2.tick_params('y', colors='k')
    # 添加图例
    fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.95), frameon=False)
    plt.grid(axis='y')
    # plt.title('')
    fig.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    figure_01(fig_path='results/fig_fund_stock_num.png')
