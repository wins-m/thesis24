"""关联网络牵引因子"""
import os
os.chdir('/home/winston/Thesis')
from typing import List
import statsmodels.api as sm
import sys
sys.path.append('/home/winston/Thesis/bin')
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from utils import get_last_period, get_next_period, data_adjopen
from data_process import data_universe
from plt_head import *


def load_w2v_sim(stime, etime) -> pd.DataFrame:
    """
    获取 stime 到 etime 范围内数据推断出来的股票 W2V 相似度（上三角矩阵）
    Return
    ------
            stockcode stockcode_right       sim    period
    0       000001.SZ       000006.SZ  0.694661  20131231
    1       000001.SZ       000009.SZ  0.693319  20131231
    2       000001.SZ       000012.SZ  0.801779  20131231
    3       000001.SZ       000021.SZ  0.568151  20131231
    4       000001.SZ       000028.SZ  0.780323  20131231 
    """
    src = f"cache/w2v.k5/sentences.10.{stime}.{etime}.word2vec.5.10.vector.30.similarity.pkl"
    if not os.path.exists(src):
        return None
    period = get_last_period(etime)
    sr: pd.Series = pd.read_pickle(src).rename('sim')
    sr.index.names = ['stockcode', 'stockcode_right']
    # sr.loc[sr < 0] = 0  # 数值范围 0~1
    # sr = (sr - sr.min()) / (sr.max() - sr.min())
    df: pd.DataFrame = sr.reset_index()
    df['period'] = period
    return df


def load_man_lnK1(period: int) -> pd.DataFrame:
    """
    获取报告期 period 建立的人工关联度（上三角矩阵）
    Return
    ------
       stockcode stockcode_right            K    period       lnK      lnK1
    0  000596.SZ       000630.SZ    51.891086  20221231  3.949147  0.702036
    1  000596.SZ       000729.SZ  4643.178428  20221231  8.443154  0.897791
    2  000596.SZ       000738.SZ   531.885128  20221231  6.276428  0.803410
    3  000596.SZ       000786.SZ   526.061043  20221231  6.265417  0.802931
    4  000596.SZ       000807.SZ    85.074528  20221231  4.443528  0.723571
    """
    src = f"cache/man/sim_Kab.{period}.pkl"
    if not os.path.exists(src):
        return None
    df = pd.read_pickle(src)
    return df


def get_2d_sim_matrix(per_df: pd.DataFrame, sim_col='sim') -> pd.DataFrame:
    """
    还原二维矩阵
    Parameters
    ----------
    per_sim: <dataframe>
           stockcode stockcode_right       sim    period
        0  000001.SZ       000006.SZ  0.694661  20131231
        1  000001.SZ       000009.SZ  0.693319  20131231
        2  000001.SZ       000012.SZ  0.801779  20131231
        3  000001.SZ       000021.SZ  0.568151  20131231
        4  000001.SZ       000028.SZ  0.780323  20131231
    Return
    ------
    df: <dataframe>
        stockcode_right  000001.SZ  000002.SZ  000006.SZ  000009.SZ  000012.SZ
        stockcode                                                             
        000001.SZ         0.000000   0.963675   0.694661   0.693319   0.801779
        000002.SZ         0.963675   0.000000   0.661964   0.585344   0.801816
        000006.SZ         0.694661   0.661964   0.000000   0.896692   0.942863
        000009.SZ         0.693319   0.585344   0.896692   0.000000   0.852964
        000012.SZ         0.801779   0.801816   0.942863   0.852964   0.000000
    """
    per_stockcodes = sorted(set(per_df['stockcode'].unique()).union(per_df['stockcode_right'].unique()))
    df = per_df.pivot(index='stockcode', columns='stockcode_right', values=sim_col).reindex(per_stockcodes, axis=1).reindex(per_stockcodes, axis=0)
    df.fillna(0, inplace=True)
    df = df.add(df.T.values)
    return df


def calculate_w2v_fvalue(cache_ar: str, cache_exp: str, prompt=False, force_update=False):
    """根据 W2V 相似度计算牵引因子值"""

    if (not force_update) and os.path.exists(cache_ar) and os.path.exists(cache_exp):
        return
    
    # 表: 复权开盘价
    adj_open = data_adjopen()
    # 每期 (月/日) 期末, 基金持仓样本股, 过去 20 日累计收益
    ret20_o2o = adj_open.pct_change(20).loc[20130101: 20230630]    
    
    # stime, etime = 20130101, 20140630
    def cal_period_exp(stime: int, etime: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """每半年度计算锚定值"""
        # 表: 每期两两关联度  - 激进: per 之后开始使用; 保守: etime 之后使用
        per_sim: pd.DataFrame = load_w2v_sim(stime, etime)
        if per_sim is None:
            return None, None
        # 所有关联股票相似度 (注意下三角矩阵转化) stocks x stocks, values in [-1, 1]
        sim_matrix = get_2d_sim_matrix(per_df=per_sim)
        # 关联股票相似度可利用的时间范围
        if prompt:  # 基金报告期末立刻知道基金持仓 (但实际上基金未进行披露)
            sper, eper = get_last_period(etime) + 1, etime
        else:  # 基金报告期半年后才建立基金持仓相似度 (符合投资者实际情况)
            sper, eper = etime + 1, get_next_period(etime)
        # 时期内超额收益 (减去截面中位值)
        r = ret20_o2o.loc[sper: eper].copy().reindex(sim_matrix.index, axis=1)
        if len(r) == 0:
            return None, None
        ar = r.sub(r.median(axis=1), axis=0)
        # 锚定值 - 个股超额收益: 原始因子
        exp = ar @ sim_matrix.T / sim_matrix.abs().sum(axis=1)
        print(exp.iloc[-2:].T.describe().T.round(3))
        return ar, exp
    # 计算牵引值
    res = []
    for st, et in zip([((fy-1)*10000+701, fy*10000+101) for fy in range(2013, 2023)],
                      [((fy-1)*10000+1231, fy*10000+630) for fy in range(2014, 2024)]):
        res.append(cal_period_exp(st[0], et[0]))
        res.append(cal_period_exp(st[1], et[1]))
    # 超额收益实现值
    fval_ar = pd.concat((_[0] for _ in res), axis=0)
    # 超额收益锚定值
    fval_exp = pd.concat((_[1] for _ in res), axis=0)

    fval_ar.to_pickle(cache_ar)
    print(cache_ar)
    fval_exp.to_pickle(cache_exp)
    print(cache_exp)


def calculate_bert_fvalue(cache_ar: str, cache_exp: str, prompt=False, force_update=False):
    """根据 BERT 相似度计算牵引因子值"""

    if (not force_update) and os.path.exists(cache_ar) and os.path.exists(cache_exp):
        return
    
    # 表: 复权开盘价
    adj_open = data_adjopen()
    # 每期 (月/日) 期末, 基金持仓样本股, 过去 20 日累计收益
    ret20_o2o = adj_open.pct_change(20).loc[20130101: 20230630]    
    
    # 所有关联股票相似度 (注意下三角矩阵转化) stocks x stocks, values in [-1, 1]
    if prompt:
        sim_matrix = pd.read_pickle('cache/bert/sim_mat.all.epoch100.ep99.pkl')
        # sim_matrix = pd.read_pickle('cache/bert/sim_mat.all.epoch100.ep40.pkl')
        # sim_matrix = pd.read_pickle('cache/bert/sim_mat.all.rpt.ep9.pkl')
        # sim_matrix = pd.read_pickle('cache/bert/bert256.similarity.pkl')
    else:
        sim_matrix = pd.read_pickle('cache/bert/bert30.20210101.20220630.similarity.pkl')
    sim_matrix -= np.eye(sim_matrix.shape[0])
    # 时期内超额收益 (减去截面中位值)
    r = ret20_o2o.loc[20140101: 20230630].copy().reindex(sim_matrix.index, axis=1)
    if len(r) == 0:
        return None, None
    ar = r.sub(r.mean(axis=1), axis=0).fillna(0)
    # 锚定值 - 个股超额收益: 原始因子
    exp = ar @ sim_matrix.T / sim_matrix.abs().sum(axis=1)
    print(exp.iloc[-2:].T.describe().T.round(3))
    # 超额收益实现值
    fval_ar = ar * r.notnull().replace(False, np.nan)
    # 超额收益锚定值
    fval_exp = exp * r.notnull().replace(False, np.nan)

    fval_ar.to_pickle(cache_ar)
    print(cache_ar)
    fval_exp.to_pickle(cache_exp)
    print(cache_exp)


def calculate_man_fvalue(cache_ar: str, cache_exp: str, prompt=False, force_update=False):
    """根据人工关联度计算牵引因子值"""

    if (not force_update) and os.path.exists(cache_ar) and os.path.exists(cache_exp):
        return
    
    # 表: 复权开盘价
    adj_open = data_adjopen()
    # 每期 (月/日) 期末, 基金持仓样本股, 过去 20 日累计收益
    ret20_o2o = adj_open.pct_change(20).loc[20130101: 20230630]    
    
    # period = 20221231
    def cal_period_exp(period: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """每半年度计算锚定值"""
        # 表: 每期两两关联度  - 激进: per 之后开始使用; 保守: etime 之后使用
        per_lnK1: pd.DataFrame = load_man_lnK1(period)
        if per_lnK1 is None:
            return None, None
        # 所有关联股票相似度 (注意下三角矩阵转化) stocks x stocks, values in [-1, 1]
        sim_matrix = get_2d_sim_matrix(per_df=per_lnK1, sim_col='lnK1')
        # 关联股票相似度可利用的时间范围
        if prompt:  # 基金报告期末立刻知道基金持仓 (但实际上基金未进行披露)
            sper, eper = period + 1, get_next_period(period)
        else:  # 基金报告期半年后才建立基金持仓相似度 (符合投资者实际情况)
            sper = get_next_period(period)
            sper, eper = sper + 1, get_next_period(sper)
        # 时期内超额收益 (减去截面中位值)
        r = ret20_o2o.loc[sper: eper].copy().reindex(sim_matrix.index, axis=1)
        if len(r) == 0:
            return None, None
        ar = r.sub(r.median(axis=1), axis=0)
        # 锚定值 - 个股超额收益: 原始因子
        exp = ar @ sim_matrix.T / sim_matrix.abs().sum(axis=1)
        print(exp.iloc[-2:].T.describe().T.round(3))
        return ar, exp
    # 计算牵引值
    res = []
    for pers in ((y*10000+630, y*10000+1231) 
                 for y in range(2013, 2023)):
        res.append(cal_period_exp(pers[0]))
        res.append(cal_period_exp(pers[1]))
    # 超额收益实现值
    fval_ar = pd.concat((_[0] for _ in res), axis=0)
    # 超额收益锚定值
    fval_exp = pd.concat((_[1] for _ in res), axis=0)

    fval_ar.to_pickle(cache_ar)
    print(cache_ar)
    fval_exp.to_pickle(cache_exp)
    print(cache_exp)


def cal_halfyear_stat(sr, br, tvr=None, ylen=240):
    """计算半年表现"""
    hy = sr.index.to_series().apply(lambda x: f"{x.year}-H{x.month//7+1}")
    grouped = sr.groupby(hy)
    res = pd.DataFrame()
    tot = pd.Series()
    res['ret'] = grouped.mean() * ylen
    tot['ret'] = sr.mean() * ylen
    # res['std'] = grouped.std() * np.sqrt(ylen)
    # tot['std'] = sr.std() * np.sqrt(ylen)
    res['sharpe'] = res['ret'] / grouped.std() / np.sqrt(ylen)
    tot['sharpe'] = tot['ret'] / sr.std() / np.sqrt(ylen)
    ar = sr - br.reindex(sr.index)
    res['aret'] = ar.groupby(hy).mean() * ylen
    tot['aret'] = ar.mean() * ylen
    mdd = sr.cumsum()
    mdd = mdd.sub(mdd.rolling(ylen, min_periods=1).max()).mul(-1)
    res['mdd'] = mdd.groupby(hy).max()
    tot['mdd'] = mdd.max()
    res['wr'] = (sr > 0).groupby(hy).mean()
    tot['wr'] = (sr > 0).mean()
    if tvr is not None:
        res['tvr'] = tvr.groupby(hy).mean()
        tot['tvr'] = tvr.mean()
    res.loc['Total'] = tot
    return res


def convert_halfyear_stat(hy_res):
    df = hy_res.copy()
    for col in ['ret', 'aret', 'mdd', 'wr', 'tvr']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f'{x*100:.1f}%')
    for col in ['sharpe']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f'{x:.2f}')
    df.index.name = '半年度'
    df = df.rename(columns={
        'ret': '年化收益',
        'sharpe': '夏普率',
        'mdd': '最大回撤',
        'aret': '年化超额收益',
        'wr': '日度胜率',
        'tvr': '日均换手率',
    }, index={'Total': '全历史'})
    # print(df)
    return df


def backtest(fval: pd.DataFrame, tab_path, fig_path=None, group_num=5,force_update=False):
    """简单回测"""
    
    # rank IC
    tab_path_ic = tab_path.replace('.xlsx', '.ic.xlsx')
    if os.path.exists(tab_path) and (os.path.exists(tab_path_ic)) and (not force_update):
        grtn: pd.DataFrame = pd.read_excel(tab_path, index_col=0)
        resic: pd.DataFrame = pd.read_excel(tab_path_ic, index_col=0, parse_dates=True)
    else:
        # 开盘价
        adjopen = data_adjopen()
        # 可交易状态
        univ0 = data_universe(kw='tradableBar5')
        univ0.index = univ0.index.strftime('%Y%m%d').astype(int)
        univ0 = univ0.reindex_like(fval)
        # 今天 (可交易时) 开盘买入, 明天开盘卖出 - 今天日间 + 今晚隔夜 的收益
        rtn1d = adjopen.pct_change().shift(-1).reindex_like(fval).mul(univ0)
        # 今天开盘买入的日收益, 昨天收盘后得到的因子值
        panel = pd.concat([rtn1d.stack().rename('rtn1d'), fval.shift(1).stack().rename('fval')], axis=1).dropna()
        # 昨天的因子值分组 (0 为因子值最大组)
        panel['group'] = panel['fval'].groupby('tradingdate').apply(lambda s: s.rank(pct=True).mul(group_num).rsub(group_num).astype(int)).values
        # 各分组日换手
        gtvr = panel.groupby('group').apply(
            lambda df: df['fval'].notnull().unstack().fillna(0).apply(
                lambda s: s / s.sum(), axis=1).diff().abs().sum(axis=1).rename('tvr')).T
        gtvr.rename(columns={gi: f'tvr_{gi}' for gi in range(group_num)}, inplace=True)
        # 各分组日收益
        grtn = panel.groupby(['tradingdate', 'group'])['rtn1d'].mean().unstack()
        # 全部基金持仓股日收益
        grtn['global'] = panel.groupby('tradingdate')['rtn1d'].mean()
        # 保存分组日收益情况
        grtn = pd.concat([grtn, gtvr], axis=1)
        grtn.to_excel(tab_path)
        print(tab_path)
        # 每日 IC
        ic = panel.groupby('tradingdate')[['rtn1d', 'fval']].corr(method='pearson')\
            .iloc[::2, 1].reset_index().set_index('tradingdate')['fval'].rename('IC')
        rnkic = panel.groupby('tradingdate')[['rtn1d', 'fval']].corr(method='spearman')\
            .iloc[::2, 1].reset_index().set_index('tradingdate')['fval'].rename('rank IC')
        cnt = panel.dropna().groupby('tradingdate')['fval'].count().rename('count') 
        resic = pd.concat([ic, rnkic, cnt], axis=1)
        resic.index = pd.to_datetime(rnkic.index, format='%Y%m%d')
        resic.to_excel(tab_path_ic)
        # print(tab_path_ic)

    sr1 = resic['rank IC'].cumsum()
    sr2 = resic['rank IC'].groupby(resic.index.strftime('%Y%m')).mean()
    sr2.index = pd.to_datetime(sr2.index, format='%Y%m')
    fig, ax1 = plt.subplots()
    ax1.bar(sr2.index, sr2.values, color='gray', alpha=.6, align='edge', width=20, label='rank IC 月均值（左轴）')
    ax1.set_ylabel('rank IC', color='gray')
    ax1.set_xlabel('日期')
    ax1.set_xlim(resic.index[0], resic.index[-1])
    ax1.grid(axis='y')
    ax2 = ax1.twinx()
    ax2.plot(sr1.index, sr1.values, color='k', linestyle='-', linewidth=1, label='rank IC 累计值（右轴）')
    ax2.set_xlim(resic.index[0], resic.index[-1])
    ax2.set_ylabel('累计 rank IC', color='k')
    mu = resic['rank IC'].mean()
    icir = mu / resic['rank IC'].std()
    annotation_text = f"均值={mu:.4f}, ICIR={icir:.4f}"
    fig.text(0.8, 0.2, annotation_text, ha='right', va='bottom', color='black')
    fig.legend(loc="upper left", bbox_to_anchor=(0.15,0.95), frameon=False)
    fig.tight_layout()
    if fig_path is not None:
        tgt = fig_path.replace('.png', '.ic.png')
        plt.savefig(tgt, transparent=True)
        print(tgt)
        plt.close()
    else:
        plt.show()

    # Portfolio Wealth
    g = grtn[range(group_num)].sub(grtn['global'], axis=0).cumsum()
    g.index = pd.to_datetime(g.index, format='%Y%m%d')
    col2name =  {0: f'多头组', group_num - 1: f'空头组'}
    col2name.update({gn: f'第 {gn+1} 组' for gn in range(1, group_num-1)})
    g = g.rename(columns=col2name)
    g.columns.name = None
    # g.index = g.index.astype(str).rename('日期')
    g.index.name = '日期'
    plt.figure()
    for i in range(group_num):
        # g.iloc[:, i].plot(linewidth=2, linestyle=('-', '--')[i%2], color='k', alpha=1 - i * (1 / group_num), label=g.columns[i])
        plt.plot(g.iloc[:, i], linewidth=2, linestyle=('-', '--')[i%2], color='k', alpha=1 - i * (1 / group_num), label=g.columns[i])
    plt.xlim(g.index[0] - pd.Timedelta(60, 'D'), g.index[-1] + pd.Timedelta(60, 'D')) 
    plt.xlabel('日期')
    plt.ylabel(f'累计相对收益率 (累加)')
    plt.legend(frameon=False, loc='lower left')
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()

    # rb, idx_name = get_csi_return(instrument='000906.SH')
    # rb, idx_name = get_csi_return(instrument='000905.SH')
    rb, idx_name = get_csi_return(instrument='000985.CSI')
    g = grtn[[0, group_num - 1]].rename(columns={0: 'H',  group_num - 1: 'L'})
    g['B'] = rb.reindex(grtn.index)
    g['H - L'] = (g['H'] - g['L']) / 2
    g['H - B'] = (g['H'] - g['B']) / 2
    g.index = pd.to_datetime(g.index, format='%Y%m%d')
    g.index.name = '日期'
    
    tvrH = grtn[f'tvr_{0}']
    tvrH.index = pd.to_datetime(tvrH.index, format='%Y%m%d')
    hyH = cal_halfyear_stat(
        sr=g['H'],
        br=g['B'],
        tvr=tvrH)
    hyHc = convert_halfyear_stat(hyH)
    tgt = tab_path.replace('.xlsx', '.hyH.xlsx')
    hyHc.to_excel(tgt)
    print(tgt)
    tvrL = grtn[f'tvr_{group_num - 1}']
    tvrL.index = pd.to_datetime(tvrL.index, format='%Y%m%d')
    hyL = cal_halfyear_stat(
        sr=g['L'],
        br=g['B'],
        tvr=tvrL)
    hyLc = convert_halfyear_stat(hyL)
    tgt = tab_path.replace('.xlsx', '.hyL.xlsx')
    hyLc.to_excel(tgt)
    print(tgt)
    hyHML = cal_halfyear_stat(
        sr=g['H - L'],
        br=g['B'],
        tvr=None)
    hyHMLc = convert_halfyear_stat(hyHML)
    tgt = tab_path.replace('.xlsx', '.hyHML.xlsx')
    hyHMLc.to_excel(tgt)
    print(tgt)
    hy_cmp = pd.concat([df.iloc[-1].rename(kw) for kw, df in {'多头组': hyHc, '空头组': hyLc, '多 - 空': hyHMLc}.items()], axis=1).T.fillna('-')
    print(hy_cmp)
    tgt = tab_path.replace('.xlsx', '.hy_cmp.xlsx')
    hy_cmp.to_excel(tgt)
    print(tgt)

    # g = g.add(1).cumprod()
    g = g.cumsum()
    plt.figure()
    plt.plot(g['H'], color='k', alpha=0.8, linewidth=2, linestyle='-', label='多头组')
    plt.plot(g['B'], color='k', alpha=0.5, linewidth=2, linestyle='-', label=idx_name)
    plt.plot(g['L'], color='k', alpha=0.2, linewidth=2, linestyle='-', label='空头组')
    plt.plot(g['H - L'], color='gray', alpha=1, linewidth=1, linestyle='-.', label='多 - 空')
    # plt.plot(g['H - B'], color='k', alpha=1, linewidth=1, linestyle='--', label=f'多头组 - {idx_name}')
    plt.xlim(g.index[0] - pd.Timedelta(60, 'D'), g.index[-1] + pd.Timedelta(60, 'D'))
    plt.xlabel('日期')
    plt.ylabel('累计收益（累加）')
    plt.grid(axis='y')
    plt.legend(frameon=False)
    plt.tight_layout()
    if fig_path is not None:
        tgt = fig_path.replace('.png', '.hml.png')
        plt.savefig(tgt, transparent=True)
        print(tgt)
        plt.close()
    else:
        plt.show()


def get_csi_return(instrument='000985.CSI'):
    """指数日度收益"""
    df = pd.read_excel('data/指数日开盘.xlsx')
    df.columns = ['instrument', 'name', 'tradingdate', 'open']
    df = df.query(f'instrument == "{instrument}"').copy()
    df['tradingdate'] = df['tradingdate'].astype(int)
    df = df.set_index('tradingdate')['open'].sort_index().pct_change().iloc[1:]
    idx_dict = {
        '000985.CSI': "中证全指",
        '932000.CSI': "中证2000",
        '000852.SH': "中证1000",
        '000905.SH': "中证500",
        '000906.SH': "中证800",
        '000300.SH': "沪深300",
    }
    return df, idx_dict[instrument]


def group_test_bert_expar():
    """BERT 相似度牵引因子"""
    calculate_bert_fvalue(
        cache_ar='cache/fval_ar.bert.pkl',
        cache_exp='cache/fval_exp.bert.pkl',
        prompt=False,
        force_update=True
    )
    fv_exp: pd.DataFrame = pd.read_pickle('cache/fval_exp.bert.pkl')
    fv_ar: pd.DataFrame = pd.read_pickle('cache/fval_ar.bert.pkl')
    fv_exp_expmar = fv_exp.sub(fv_ar.mul(fv_ar.pow(2).sum(1).rdiv(1).mul(fv_ar.mul(fv_exp).sum(1)), axis=0))
    backtest(fv_exp_expmar.loc[20140701: 20230630], group_num=5,
             fig_path='results/bert_expmar_g5.png',
             tab_path='results/bert_expmar_g5.xlsx',
             force_update=True)
  

def group_test_expar(gn=5, neu=None, prp=False, fu=False):
    """W2V相似牵引因子 - 保守"""
    fv_path = f"cache/fval_w2v_expmar{('', '.prp')[prp]}{(f'.{neu}', '')[neu is None]}.pkl"
    if os.path.exists(fv_path) and (not fu):
        fv = pd.read_pickle(fv_path)
    else:
        calculate_w2v_fvalue(
            cache_ar=f'cache/fval_ar{("", ".prp")[prp]}.pkl',
            cache_exp=f'cache/fval_exp{("", ".prp")[prp]}.pkl',
            prompt=prp,
            force_update=fu
        )
        fv_exp: pd.DataFrame = pd.read_pickle(f'cache/fval_exp{("", ".prp")[prp]}.pkl')
        fv_ar: pd.DataFrame = pd.read_pickle(f'cache/fval_ar{("", ".prp")[prp]}.pkl')
        fv_w2v_expmar = fv_exp.sub(fv_ar.mul(fv_ar.pow(2).sum(1).rdiv(1).mul(fv_ar.mul(fv_exp).sum(1)), axis=0))
        fv = neutralize_df(fv=fv_w2v_expmar, neu=neu)
        fv.to_pickle(fv_path)
        print(fv_path)
    backtest(fv.loc[20140701: 20230630], group_num=gn,
             fig_path=f'results/w2v_expmar{("", "_prp")[prp]}_g{gn}{"."+neu if neu else ""}.png',
             tab_path=f'results/w2v_expmar{("", "_prp")[prp]}_g{gn}{"."+neu if neu else ""}.xlsx',
             force_update=fu)


def reg_resid(y: pd.Series, x: List[pd.Series], add_const=True) -> pd.Series:
    pn = pd.concat([y] + x, axis=1)
    pn = pn.dropna()  # 只要有缺失都不加入回归
    y, X = pn.iloc[:, 0], pn.iloc[:, 1:]
    if add_const:
        X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.resid


def cross_section_regr_resid(y: pd.DataFrame, X: List[pd.DataFrame], add_const=False):
    res = pd.concat(
        [reg_resid(y=y.loc[per], x=[x.loc[per] 
                                    for x in X if x.loc[per].abs().sum() > 0], add_const=add_const).rename(per)
         for per in y.index], axis=1).T
    res.index.name = 'tradingdate'
    return res


def group_test_man_expar(gn=5, prp=False, fu=False, neu=None):
    """人工关联度牵引因子"""
    fv_path = f"cache/fval_man_expmar{('', '.prp')[prp]}{(f'.{neu}', '')[neu is None]}.pkl"
    if os.path.exists(fv_path) and (not fu):
        fv = pd.read_pickle(fv_path)
    else:
        calculate_man_fvalue(
            cache_ar=f"cache/fval_manar{('', '.prp')[prp]}.pkl",
            cache_exp=f"cache/fval_manexp{('', '.prp')[prp]}.pkl",
            prompt=prp,
            force_update=fu
        )
        fv_exp: pd.DataFrame = pd.read_pickle(f"cache/fval_manexp{('', '.prp')[prp]}.pkl")
        fv_ar: pd.DataFrame = pd.read_pickle(f"cache/fval_manar{('', '.prp')[prp]}.pkl")
        fv_man_expmar = fv_exp.sub(fv_ar.mul(fv_ar.pow(2).sum(1).rdiv(1).mul(fv_ar.mul(fv_exp).sum(1)), axis=0))
        fv = neutralize_df(fv=fv_man_expmar, neu=neu)
        fv.to_pickle(fv_path)
        print(fv_path)
    backtest(fv.loc[20140701: 20230630], group_num=gn,
             fig_path=f'results/man_expmar{("", "_prp")[prp]}_g{gn}{"."+neu if neu else ""}.png',
             tab_path=f'results/man_expmar{("", "_prp")[prp]}_g{gn}{"."+neu if neu else ""}.xlsx',
             force_update=fu)


def neutralize_df(fv: pd.DataFrame, neu) -> pd.DataFrame:
    if neu is not None:
        if 'i' in neu:
            from utils import get_industry
            indus = get_industry(stime=fv.index[0], etime=fv.index[-1])
            indus = indus.unstack().reindex_like(fv)
            pn = fv.stack().rename('fv')
            pn = pd.concat([pn, indus.stack().rename('ind').reindex(pn.index)], axis=1)
            pn = pn.reset_index()
            indm = pn.groupby(['tradingdate', 'ind'])['fv'].mean().rename('indm')
            pn = pn.merge(indm, on=['tradingdate', 'ind'], how='outer')
            pn.set_index(['tradingdate', 'stockcode'], inplace=True)
            fv = (pn['fv'] - pn['indm']).unstack()
        if 'v' in neu:
            stdlnmv = get_stdlnmv().reindex_like(fv).fillna(method='ffill')
            fv = cross_section_regr_resid(y=fv, X=[stdlnmv], add_const=True)
    return fv


def get_stdlnmv():
    """自由流通市值-对数-标准化"""
    mv = pd.read_pickle("data/marketcap_freefloat.pkl")
    mv.index = mv.index.strftime('%Y%m%d').astype(int)
    lnmv = mv.applymap(np.log)
    stdlnmv = lnmv
    # stdlnmv = lnmv.sub(lnmv.mean(axis=1), axis=0).div(lnmv.std(axis=1).replace(0, 1), axis=0)
    return stdlnmv


def group_test_no_beta_reg():
    """去除自身 AR 和 beta"""
    pass


def group_test_compare_all():
    res = []
    for file in (
        "results/man_expmar_g5.hy_cmp.xlsx",
        "results/man_expmar_g5.i.hy_cmp.xlsx",
        "results/man_expmar_g5.iv.hy_cmp.xlsx",
        "results/w2v_expmar_g5.hy_cmp.xlsx",
        "results/w2v_expmar_g5.i.hy_cmp.xlsx",
        "results/w2v_expmar_g5.iv.hy_cmp.xlsx",
    ):
        from pathlib import Path
        f = Path(file)
        df = pd.read_excel(f, index_col=0)
        print(df)
        df.index.name = '分组'
        kw = decode_filename(f.name)
        df.loc[kw] = None
        df = df.loc[[kw, '多头组', '多 - 空', '空头组']]
        res.append(df)
    res = pd.concat(res, axis=0)
    print(res)
    tgt = 'results/all.hy_cmp.xlsx'
    res.to_excel(tgt)
    print(tgt)
    tgt = 'results/all.hy_cmp.H.xlsx'
    res.query("分组 == '多头组'").T.to_excel(tgt)
    print(tgt)
    tgt = 'results/all.hy_cmp.HML.xlsx'
    res.query("分组 == '多 - 空'").T.to_excel(tgt)
    print(tgt)
    tgt = 'results/all.hy_cmp.L.xlsx'
    res.query("分组 == '空头组'").T.to_excel(tgt)
    print(tgt)


def decode_filename(fn: str) -> str:
    ns = fn.split('.')
    dic0 = {
        'man': '人工关联度牵引因子',
        'w2v': 'W2V相似度牵引因子',
    }
    dic1 = {
        'ic': '',
        'hy_cmp': '',
        'i': '（行业中性）',
        'iv': '（行业、市值中性）',
    }
    return dic0[ns[0].split('_')[0]] + dic1[ns[1]]

    
def group_test_compare_ic_all():
    res = []    
    for file in [
        "results/man_expmar_g5.ic.xlsx",
        "results/man_expmar_g5.i.ic.xlsx",
        "results/man_expmar_g5.iv.ic.xlsx",
        "results/w2v_expmar_g5.ic.xlsx",
        "results/w2v_expmar_g5.i.ic.xlsx",
        "results/w2v_expmar_g5.iv.ic.xlsx",
    ]:
        f0 = Path(file.replace('iv', 'hy_cmp'))
        f = Path(file)
        df = pd.read_excel(f, index_col=0, parse_dates=True)
        # print(df)
        re = pd.Series(name=decode_filename(f.name))
        # re.loc['IC均值'] = df['IC'].mean()
        # re.loc['IC IR'] = df['IC'].mean() / df['IC'].std()
        re.loc['rank IC均值'] = f"{df['rank IC'].mean()*100:.2f}%"
        re.loc['rank ICIR'] = f"{df['rank IC'].mean() / df['rank IC'].std():.4f}"
        re.loc['日均样本量'] = f"{df['count'].mean():.1f}"
        # print(re)
        res.append(re)
    res = pd.concat(res, axis=1)
    res = res.T
    tgt = 'results/iv_compare_all.xlsx'
    res.to_excel(tgt)
    print(tgt)


def fama_macbeth_barra():
    fv = pd.concat([pd.read_pickle(src).stack().rename(Path(src).name.replace('fval_', '')) 
                    for src in ("cache/fval_man_expmar.pkl", "cache/fval_w2v_expmar.pkl",
                                "cache/fval_man_expmar.i.pkl", "cache/fval_w2v_expmar.i.pkl")],
                   axis=1)
    tgt = 'cache/fval_.h5'
    fv.stack().to_hdf(tgt, key='data', complevel=9, index=False)
    tgt = 'cache/fval_index.h5'
    pd.Series(index=fv.index).to_hdf(tgt, key='data', complevel=9, index=False)



if __name__ == '__main__':
    gn = 5
    neu = None
    # group_test_man_expar(gn=gn, prp=False, fu=True, neu=neu)
    # group_test_expar(gn=gn, prp=False, fu=True, neu=neu)
    # neu = 'i'
    # group_test_man_expar(gn=gn, prp=False, fu=True, neu=neu)
    # group_test_expar(gn=gn, prp=False, fu=True, neu=neu)
    # neu = 'iv'
    # group_test_man_expar(gn=gn, prp=False, fu=True, neu=neu)
    # group_test_expar(gn=gn, prp=False, fu=True, neu=neu)

    # group_test_bert_expar()

    # group_test_compare_all()
    group_test_compare_ic_all()
