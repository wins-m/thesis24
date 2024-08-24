"""
评估嵌入表示的有效性.
1. 聚类的有效性: 关于行业/风格的可视化
2. 对收益率 (协方差) 相似度预测的有效性; 和线性模型的对比
3. 对预测公募后续持仓的有效性

输出: 表格, 图像
"""
import os
import numpy as np
import pandas as pd
from utils import get_industry, get_last_period
from data_process import data_mv, data_asset_mv
from plt_head import *
_PATH = "/home/winston/Thesis"
os.chdir(_PATH)


def w2v_sim_indus_comp(tab_path=None, fig_path=None, force_update=False):
    """同行业/不同行业的 Word2Vec 相似度均值对比"""

    def per_sim_comp(stime, etime):
        """Word2Vec计算的相似度，增加行业类别标签"""
        src = f'cache/sentences.10.{stime}.{etime}.word2vec.5.10.vector.30.similarity.pkl'
        if not os.path.exists(src):
            print(src)
            return None
        period = get_last_period(etime)

        # 两两关联度
        similarities: pd.Series = pd.read_pickle(src).rename('sim')
        similarities.index.names = ['stockcode', 'stockcode_right']
        similarities = similarities.reset_index()
        similarities['period'] = period
        # >>> similarities
        #         stockcode stockcode_right       sim    period
        # 0       000001.SZ       000006.SZ  0.723740  20140630
        # 1       000001.SZ       000009.SZ  0.631942  20140630
        # 2       000001.SZ       000012.SZ  0.582789  20140630
        # 3       000001.SZ       000021.SZ  0.326600  20140630
        # 4       000001.SZ       000026.SZ  0.581712  20140630

        # 行业类别
        ind = get_industry(stime=period//100*100+1, etime=period, lvl=1)
        ind = ind.loc[ind.index[-1][0]].rename('indus1')
        ind = ind.reset_index()
        ind.columns = ['stockcode', 'indus1']

        sim_indus = similarities\
            .merge(ind, on=['stockcode'], how='left')\
            .merge(ind.rename(columns={'stockcode': 'stockcode_right', 'indus1': 'indus1_right'}), on=['stockcode_right'], how='left')
        sim_indus['iden_indus'] = sim_indus['indus1'].eq(sim_indus['indus1_right']).astype(int)
        # >>> print(sim_indus)
        #         stockcode stockcode_right       sim    period  indus1  indus1_right  iden_indus
        # 0       000001.SZ       000006.SZ  0.723740  20140630      21            23           0
        # 1       000001.SZ       000009.SZ  0.631942  20140630      21            23           0
        # 2       000001.SZ       000012.SZ  0.582789  20140630      21             8           0
        # 3       000001.SZ       000021.SZ  0.326600  20140630      21            25           0
        # 4       000001.SZ       000026.SZ  0.581712  20140630      21             9           0

        res = sim_indus.groupby('iden_indus')['sim'].describe()
        res = res.reset_index()
        res['period'] = period
        res = res.set_index(['period', 'iden_indus'])
        print(res)
        return res
    
    if (tab_path is None) or (not os.path.exists(tab_path)) or force_update:
        res = {}
        for st, et in zip(
            [((fy-1)*10000+701, fy*10000+101) for fy in range(2013, 2023)],
            [((fy-1)*10000+1231, fy*10000+630) for fy in range(2014, 2024)]
        ):
            res[get_last_period(et[0])] = per_sim_comp(st[0], et[0])
            res[get_last_period(et[1])] = per_sim_comp(st[1], et[1])
    
        tab = pd.concat(res.values())
        if tab_path is not None:
            tab.to_excel(tab_path)
    else:
        tab = pd.read_excel(tab_path, index_col=[0, 1])

    g = tab['mean'].unstack().rename(columns={0: '不同行业', 1: '同行业'})
    g.columns.name = None
    g.index = g.index.astype(str).rename('相似度均值')
    
    plt.figure()
    plt.plot(g.index, g['同行业'], label='同行业', linewidth=3, color='k')
    plt.plot(g.index, g['不同行业'], label='不同行业', linewidth=3, color='gray', linestyle='--')
    plt.xticks(rotation=45, ha='right', va='top')
    plt.xlabel('报告期')
    plt.ylim(0, 1.0)
    plt.ylabel('平均相似度')
    plt.legend(frameon=False)
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


def w2v_sim_barra_comp(fname='MV', kw='市值', group_num=5, tab_path=None, fig_path=None, force_update=False):
    """不同风格分组 Word2Vec 相似度均值对比"""

    fval = pd.read_hdf(f"data/QUANTILE_{fname}.h5", key='data')
    # >>> fval.head()
    # date      instrument
    # 20100104  000001.SZ     0
    #           000002.SZ     0
    #           000004.SZ     9
    #           000005.SZ     2
    #           000006.SZ     2
    # Name: QUANTILE_MV, dtype: int64

    def per_sim_comp(stime, etime):
        """Word2Vec计算的相似度，增加风格类别标签"""
        src = f'cache/sentences.10.{stime}.{etime}.word2vec.5.10.vector.30.similarity.pkl'
        if not os.path.exists(src):
            return None
        period = get_last_period(etime)
        # 两两关联度
        similarities: pd.Series = pd.read_pickle(src).rename('sim')
        similarities.index.names = ['stockcode', 'stockcode_right']
        similarities = similarities.reset_index()
        similarities['period'] = period
        # 两两各自的风格类别
        fv_per = fval.loc[fval.loc[:period].index[-1][0]] // 2
        similarities[fname] = fv_per.reindex(similarities['stockcode']).values
        similarities[fname + '_right'] = fv_per.reindex(similarities['stockcode_right']).values
        # 风格类别相同
        similarities[f'iden_{fname}'] = similarities[fname].eq(similarities[f'{fname}_right']).astype(int)
        # >>> similarities.head()
        #    stockcode stockcode_right       sim    period  MV_right   MV  iden_MV
        # 0  000001.SZ       000006.SZ  0.694661  20131231       2.0  0.0        0
        # 1  000001.SZ       000009.SZ  0.693319  20131231       1.0  0.0        0
        # 2  000001.SZ       000012.SZ  0.801779  20131231       1.0  0.0        0
        # 3  000001.SZ       000021.SZ  0.568151  20131231       1.0  0.0        0
        # 4  000001.SZ       000028.SZ  0.780323  20131231       1.0  0.0        0

        # 相同风格分组下的关联度指标统计量
        res = similarities.query(f"iden_{fname} == 1").groupby(fname)['sim'].describe()
        res = res.reset_index()
        res[fname] = res[fname].astype(int)
        res['count'] = res['count'].astype(int)
        res['period'] = period
        res = res.set_index(['period', fname])
        print(res)
        return res

    if (tab_path is None) or (not os.path.exists(tab_path)) or force_update:
        res = {}
        for st, et in zip(
            [((fy-1)*10000+701, fy*10000+101) for fy in range(2013, 2023)],
            [((fy-1)*10000+1231, fy*10000+630) for fy in range(2014, 2024)]
        ):
            # stime, etime = 20130101, 20140630
            res[get_last_period(et[0])] = per_sim_comp(st[0], et[0])
            res[get_last_period(et[1])] = per_sim_comp(st[1], et[1])
    
        tab = pd.concat(res.values(), axis=0)
        if tab_path is not None:
            tab.to_excel(tab_path)
    else:
        tab = pd.read_excel(tab_path, index_col=[0, 1])

    g = tab['mean'].unstack().rename(columns={0: f'高{kw}', 1: '第二组',
                                              2: '第三组', 3: '第四组',
                                              4: f'低{kw}'})
    g.columns.name = None
    g.index = g.index.astype(str).rename('报告期')

    plt.figure()
    for i in range(5):
        plt.plot(g.index, g.iloc[:, i], linewidth=3, linestyle=('-', '--')[i%2], color='k', alpha=1 - i * (1 / group_num), label=g.columns[i])
    plt.xticks(rotation=45, ha='right', va='top')
    plt.xlabel('报告期')
    plt.ylim(0, 1.0)
    plt.ylabel('平均相似度')
    plt.legend(frameon=False)
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


def w2v_sim_mv_comp(tab_path=None, fig_path=None, group_num=5, kind=1, amv=False, force_update=False):
    """不同市值分组下的 Word2Vec 相似度对比"""

    if amv:
        mv = data_asset_mv()
        mv.index = mv.index.strftime('%Y%m%d').astype(int)
        fname = 'amv'
    else:
        mv = data_mv()
        mv.index = mv.index.strftime('%Y%m%d').astype(int)
        fname = 'mv'

    # stime, etime = 20200101, 20210630
    def per_sim_comp(stime, etime):
        """Word2Vec计算的相似度，增加风格类别标签"""
        src = f'cache/sentences.10.{stime}.{etime}.word2vec.5.10.vector.30.similarity.pkl'
        if not os.path.exists(src):
            print(src)
            return None
        period = get_last_period(etime)
        # 两两关联度
        similarities: pd.Series = pd.read_pickle(src).rename('sim')
        similarities.index.names = ['stockcode', 'stockcode_right']
        similarities = similarities.reset_index()
        similarities['period'] = period
        # 两两各自的风格类别
        if kind == 1:
            fv_per = mv.loc[mv.loc[:period].index[-1]].reindex(similarities['stockcode'].unique())\
                .rank(pct=True).mul(group_num).rsub(group_num).fillna(-1).astype(int).replace(-1, None)
        elif kind == 2:
            fv_per = mv.loc[mv.loc[:period].index[-1]]\
                .rank(pct=True).mul(group_num).rsub(group_num).fillna(-1).astype(int).replace(-1, None)\
                .reindex(similarities['stockcode'].unique())
        similarities[fname] = fv_per.reindex(similarities['stockcode']).values
        similarities[fname + '_right'] = fv_per.reindex(similarities['stockcode_right']).values
        # 风格类别相同
        similarities[f'iden_{fname}'] = similarities[fname].eq(similarities[f'{fname}_right']).astype(int)

        # 相同风格分组下的关联度指标统计量
        res = similarities.query(f"iden_{fname} == 1").groupby(fname)['sim'].describe()
        res = res.reset_index()
        res[fname] = res[fname].astype(int)
        res['count'] = res['count'].astype(int)
        res['period'] = period
        res = res.set_index(['period', fname])
        print(res)
        return res
    
    if (tab_path is None) or (not os.path.exists(tab_path)) or force_update:
        res = {}
        for st, et in zip(
            [((fy-1)*10000+701, fy*10000+101) for fy in range(2013, 2023)],
            [((fy-1)*10000+1231, fy*10000+630) for fy in range(2014, 2024)]
        ):
            # stime, etime = 20130101, 20140630
            res[get_last_period(et[0])] = per_sim_comp(st[0], et[0])
            res[get_last_period(et[1])] = per_sim_comp(st[1], et[1])
    
        tab = pd.concat(res.values(), axis=0)
        if tab_path is not None:
            tab.to_excel(tab_path)
    else:
        tab = pd.read_excel(tab_path, index_col=[0, 1])

    col2name =  {0: f'大市值', group_num - 1: f'小市值'}
    col2name.update({gn: f'第 {gn+1} 组' for gn in range(1, group_num-1)})
    g = tab['mean'].unstack().rename(columns=col2name)
    g.columns.name = None
    g.index = g.index.astype(str).rename('报告期')

    plt.figure()
    for i in range(group_num):
        if i < g.shape[1]:
            plt.plot(g.index, g.iloc[:, i], linewidth=3, linestyle=('-', '--')[i%2], color='k', alpha=1 - i * (1 / group_num), label=g.columns[i])
    plt.xticks(rotation=45, ha='right', va='top')
    plt.xlabel('报告期')
    plt.ylim(0, 1.0)
    plt.ylabel('平均相似度')
    plt.legend(frameon=False)
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


def w2v_ret_r2(tab_path=None, fig_path=None, group_num=5, sample_n=1000, force_update=False):
    """根据相似度分组，组内报告期后 60 日收益率相关性系数(平方)的均值"""
    
    adjopen = pd.read_hdf('data/复权开盘价.h5', key='data')
    ret_o2o = adjopen.pct_change().loc[20130101: 20230701]
    ret_o2o.index.name = 'tradingdate'

    # stime, etime = 20200101, 20210630
    def per_sim_comp(stime, etime):
        """每个报告期单独处理"""

        src = f'cache/sentences.10.{stime}.{etime}.word2vec.5.10.vector.30.similarity.pkl'
        if not os.path.exists(src):
            print(src)
            return None
        period = get_last_period(etime)
        print(period)
        # 两两关联度
        similarities: pd.Series = pd.read_pickle(src).rename('sim')
        similarities.index.names = ['stockcode', 'stockcode_right']
        similarities = similarities.reset_index()
        similarities['period'] = period
        similarities['sim_g'] = similarities['sim'].rank(pct=True).mul(group_num).rsub(group_num).astype(int)
        # 报告期后紧接的 60 日收益率 (复权开盘价) 
        ret60d = ret_o2o.query(f'tradingdate > {period}').iloc[:60].dropna(axis=1)
        # 两两 60 日收益相关性系数绝对值统计量
        res = [] 
        # 样本太多了, 等间隔抽样 sample_n 对
        for gn in range(0, group_num):
            print('Similarity group:', gn)
            if gn == -1:  # 所有基金持仓样本股内统计
                df = similarities
            else:  # 相似度分组内抽样
                df = similarities.query(f'sim_g == {gn}')
            df = df[['stockcode', 'stockcode_right']]
            g_res = {}
            gap = max(1, len(df) // sample_n)
            for i in range(0, len(df) - gap, gap):
                stk1, stk2 = df.iloc[i].values
                idelta = 0
                while ((stk1 not in ret60d.columns) or (stk2 not in ret60d.columns)) and (idelta < gap):
                    idelta += 1
                    stk1, stk2 = df.iloc[i + idelta].values
                val = ret60d[stk1].corr(ret60d[stk2])
                while np.isnan(val) and (idelta < gap):
                    idelta += 1
                    stk1, stk2 = df.iloc[i + idelta].values
                    while ((stk1 not in ret60d.columns) or (stk2 not in ret60d.columns)) and (idelta < gap):
                        idelta += 1
                        stk1, stk2 = df.iloc[i + idelta].values
                    val = ret60d[stk1].corr(ret60d[stk2])
                if not np.isnan(val):
                    g_res[(stk1, stk2)] = val
                else:
                    print(stk1, stk2, end='\r')
            gres = pd.Series(g_res, name='corr')
            gres.index.names = ['stockcode', 'stockcode_right']
            gres = gres.reset_index()
            gres['group'] = gn
            gres = gres.set_index(['group', 'stockcode', 'stockcode_right'])
            res.append(gres.iloc[-sample_n:]['corr'])
        res = pd.concat(res, axis=0)
        res1 = res.abs().groupby('group').describe()
        print(res1)
        res1 = res1.reset_index()
        res1['period'] = period
        return res1.set_index(['period', 'group'])

    if (tab_path is None) or (not os.path.exists(tab_path)) or force_update:
        res = {}
        for st, et in zip([((fy-1)*10000+701, fy*10000+101) for fy in range(2013, 2023)],
                          [((fy-1)*10000+1231, fy*10000+630) for fy in range(2014, 2024)]):
            # stime, etime = 20130101, 20140630
            res[get_last_period(et[0])] = per_sim_comp(st[0], et[0])
            res[get_last_period(et[1])] = per_sim_comp(st[1], et[1])
    
        tab = pd.concat(res.values(), axis=0)
        if tab_path is not None:
            tab.to_excel(tab_path)
    else:
        tab = pd.read_excel(tab_path, index_col=[0, 1])

    col2name =  {-1: '随机抽样', 0: f'高相似度', group_num - 1: f'低相似度'}
    col2name.update({gn: f'第 {gn+1} 组' for gn in range(1, group_num-1)})
    g = tab['mean'].unstack().rename(columns=col2name)
    g.columns.name = None
    g.index = g.index.astype(str).rename('报告期')
    # gm = tab['mean'].groupby('period').mean()
    # g1 = g.sub(gm.values, axis=0)
    # g1 = g.sub(g.iloc[:, group_num // 2], axis=0)
    g1 = g.sub(g.mean(axis=1), axis=0)
    plt.figure()
    for i in range(group_num):
        plt.plot(g1.index, g1.iloc[:, i], linewidth=3, linestyle=('-', '--')[i%2], color='k', alpha=1 - i * (1 / group_num), label=g.columns[i])
    plt.xticks(rotation=45, ha='right', va='top')
    plt.xlabel('报告期')
    # plt.ylim(0, 1.0)
    plt.ylabel(f'相关性系数绝对均值差异')
    plt.legend(frameon=False, loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


def man_ret_r2(tab_path=None, fig_path=None, group_num=5, sample_n=1000, force_update=False):
    """根据 lnK1 关联度分组，组内报告期后 60 日收益率相关性系数(平方)的均值"""
    
    adjopen = pd.read_hdf('data/复权开盘价.h5', key='data')
    ret_o2o = adjopen.pct_change().loc[20130101: 20230701]
    ret_o2o.index.name = 'tradingdate'

    def per_comp(per):
        """每个报告期单独处理"""
        src = f"cache/sim_Kab.{per}.pkl"
        if not os.path.exists(src):
            print(src)
            return None
        df = pd.read_pickle(src)
        if 'lnK1' not in df.columns:
            df = df[df.stockcode.apply(lambda x: x[:6]) < df.stockcode_right.apply(lambda x: x[:6])]
            if 'lnK' not in df.columns:
                df['lnK'] = df.K.apply(np.log)  # 取对数
            df['lnK1'] = df['lnK'].sub(df['lnK'].min()).div(df['lnK'].max() - df['lnK'].min())
            df.dropna(inplace=True)
            df.to_pickle(src); print('update:', src)
        similarities = df
        similarities['sim_g'] = similarities['lnK1'].rank(pct=True).mul(group_num).rsub(group_num).astype(int)
        # >>> similarities
        #          stockcode stockcode_right            K    period       lnK      lnK1   sim_g
        # 0        000596.SZ       000630.SZ    51.891086  20221231  3.949147  0.345447       1
        # 1        000596.SZ       000729.SZ  4643.178428  20221231  8.443154  0.720609       0
        # 2        000596.SZ       000738.SZ   531.885128  20221231  6.276428  0.539730       0
        # 3        000596.SZ       000786.SZ   526.061043  20221231  6.265417  0.538810       0
        # 4        000596.SZ       000807.SZ    85.074528  20221231  4.443528  0.386718       0
        # ...            ...             ...          ...       ...       ...       ...     ...

        # 报告期后紧接的 60 日收益率 (复权开盘价) 
        ret60d = ret_o2o.query(f'tradingdate > {per}').iloc[:60].dropna(axis=1)

        # 两两 60 日收益相关性系数绝对值统计量
        res = [] 
        # 样本太多了, 等间隔抽样 sample_n 对
        for gn in range(0, group_num):
            print('Similarity group:', gn)
            if gn == -1:  # 所有基金持仓样本股内统计
                df = similarities
            else:  # 相似度分组内抽样
                df = similarities.query(f'sim_g == {gn}')
            df = df[['stockcode', 'stockcode_right']]
            g_res = {}
            gap = max(1, len(df) // sample_n)
            for i in range(0, len(df) - gap, gap):
                stk1, stk2 = df.iloc[i].values
                idelta = 0
                while ((stk1 not in ret60d.columns) or (stk2 not in ret60d.columns)) and (idelta < gap):
                    idelta += 1
                    stk1, stk2 = df.iloc[i + idelta].values
                val = ret60d[stk1].corr(ret60d[stk2])
                while np.isnan(val) and (idelta < gap):
                    idelta += 1
                    stk1, stk2 = df.iloc[i + idelta].values
                    while ((stk1 not in ret60d.columns) or (stk2 not in ret60d.columns)) and (idelta < gap):
                        idelta += 1
                        stk1, stk2 = df.iloc[i + idelta].values
                    val = ret60d[stk1].corr(ret60d[stk2])
                if not np.isnan(val):
                    g_res[(stk1, stk2)] = val
                else:
                    print(stk1, stk2, end='\r')
            gres = pd.Series(g_res, name='corr')
            gres.index.names = ['stockcode', 'stockcode_right']
            gres = gres.reset_index()
            gres['group'] = gn
            gres = gres.set_index(['group', 'stockcode', 'stockcode_right'])
            res.append(gres.iloc[-sample_n:]['corr'])
        res = pd.concat(res, axis=0)

        res1 = res.abs().groupby('group').describe()
        res1 = res1.reset_index()
        res1['period'] = per
        res1 = res1.set_index(['period', 'group'])
        print(res1)

        return res1

    if (tab_path is None) or (not os.path.exists(tab_path)) or force_update:
        res = {}
        for pers in ((y*10000+630, y*10000+1231) 
                     for y in range(2013, 2023)):
            res[pers[0]] = per_comp(pers[0])
            res[pers[1]] = per_comp(pers[1])
        tab = pd.concat(res.values())
        if tab_path is not None:
            tab.to_excel(tab_path)
            print(tab_path)
    else:
        tab = pd.read_excel(tab_path, index_col=[0, 1])
    
    col2name =  {-1: '随机抽样', 0: f'高关联度'}
    col2name.update({gn: f'第 {gn+1} 组' for gn in range(1, group_num-1)})
    col2name.update({group_num - 1: f'低关联度'})
    g = tab['mean'].unstack().rename(columns=col2name)
    g.columns.name = None
    g.index = g.index.astype(str).rename('报告期')
    # g1 = g.sub(g.iloc[:, group_num // 2], axis=0)
    g1 = g.sub(g.mean(axis=1), axis=0)
    plt.figure()
    for i in range(group_num):
        plt.plot(g1.index, g1.iloc[:, i], linewidth=3, linestyle=('-', '--')[i%2], color='k', alpha=1 - i * (1 / group_num), label=g.columns[i])
    plt.xticks(rotation=45, ha='right', va='top')
    plt.xlabel('报告期')
    # plt.ylim(0, 1.0)
    plt.ylabel(f'相关性系数绝对均值差异')
    plt.legend(frameon=False, loc='lower right')
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


def man_lnK1_indus_comp(tab_path=None, fig_path=None, force_update=False):
    """同行业/不同行业的 人工关联度 均值对比"""

    # per = 20221231
    def per_comp(per):
        """比较一个报告期"""
        src = f"cache/sim_Kab.{per}.pkl"
        if not os.path.exists(src):
            print(src)
            return None
        df = pd.read_pickle(src)
        # >>> df
        #          stockcode stockcode_right            K    period
        # 0        000596.SZ       000630.SZ    51.891086  20221231
        # 1        000596.SZ       000729.SZ  4643.178428  20221231
        # 2        000596.SZ       000738.SZ   531.885128  20221231
        # 3        000596.SZ       000786.SZ   526.061043  20221231
        # 4        000596.SZ       000807.SZ    85.074528  20221231
        if 'lnK1' not in df.columns:
            df = df[df.stockcode.apply(lambda x: x[:6]) < df.stockcode_right.apply(lambda x: x[:6])]
            if 'lnK' not in df.columns:
                df['lnK'] = df.K.apply(np.log)  # 取对数
            df['lnK1'] = df['lnK'].sub(df['lnK'].min()).div(df['lnK'].max() - df['lnK'].min())
            df.dropna(inplace=True)
            df.to_pickle(src); print('update:', src)
        
        # 行业类别
        ind = get_industry(stime=per//100*100+1, etime=per, lvl=1)
        ind = ind.loc[ind.index[-1][0]].rename('indus1')
        ind = ind.reset_index()
        ind.columns = ['stockcode', 'indus1']

        # 关联度 + 左右行业类别
        sim_indus = df\
            .merge(ind, on=['stockcode'], how='left')\
            .merge(ind.rename(columns={'stockcode': 'stockcode_right', 'indus1': 'indus1_right'}),
                   on=['stockcode_right'], how='left')
        sim_indus['iden_indus'] = sim_indus['indus1'].eq(sim_indus['indus1_right']).astype(int)
        # >>> sim_indus
        #          stockcode stockcode_right            K    period       lnK      lnK1  indus1  indus1_right  iden_indus
        # 0        000596.SZ       000630.SZ    51.891086  20221231  3.949147  0.345447      19             3           0
        # 1        000596.SZ       000729.SZ  4643.178428  20221231  8.443154  0.720609      19            19           1
        # 2        000596.SZ       000738.SZ   531.885128  20221231  6.276428  0.539730      19            12           0
        # 3        000596.SZ       000786.SZ   526.061043  20221231  6.265417  0.538810      19             8           0
        # 4        000596.SZ       000807.SZ    85.074528  20221231  4.443528  0.386718      19             3           0
        # ...            ...             ...          ...       ...       ...       ...     ...           ...         ...

        res = sim_indus.groupby('iden_indus')['lnK1'].describe()
        res = res.reset_index()
        res['period'] = per
        res = res.set_index(['period', 'iden_indus'])
        print(res)

        return res

    if (tab_path is None) or (not os.path.exists(tab_path)) or force_update:
        res = {}
        for pers in ((y*10000+630, y*10000+1231) 
                     for y in range(2013, 2023)):
            res[pers[0]] = per_comp(pers[0])
            res[pers[1]] = per_comp(pers[1])
        tab = pd.concat(res.values())
        if tab_path is not None:
            tab.to_excel(tab_path)
            print(tab_path)
    else:
        tab = pd.read_excel(tab_path, index_col=[0, 1])

    g = tab['mean'].unstack().rename(columns={0: '不同行业', 1: '同行业'})
    g.columns.name = None
    g.index = g.index.astype(str).rename('关联度均值')
    plt.figure()
    plt.plot(g.index, g['同行业'], label='同行业', linewidth=3, color='k')
    plt.plot(g.index, g['不同行业'], label='不同行业', linewidth=3, color='gray', linestyle='--')
    plt.xticks(rotation=45, ha='right', va='top')
    plt.xlabel('报告期')
    plt.ylim(0, 1.0)
    plt.ylabel('平均关联度')
    plt.legend(frameon=False)
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


def man_sim_mv_comp(tab_path=None, fig_path=None, group_num=5, kind=1, amv=False, force_update=False):
    """不同市值分组下的 人工关联度 对比"""

    if amv:
        mv = data_asset_mv()
        mv.index = mv.index.strftime('%Y%m%d').astype(int)
        fname = 'amv'
    else:
        mv = data_mv()
        mv.index = mv.index.strftime('%Y%m%d').astype(int)
        fname = 'mv'

    # period = 20221231
    def per_comp(period):
        """比较一个报告期"""
        src = f"cache/sim_Kab.{period}.pkl"
        if not os.path.exists(src):
            print(src)
            return None
        df = pd.read_pickle(src)
        # >>> df
        #          stockcode stockcode_right            K    period
        # 0        000596.SZ       000630.SZ    51.891086  20221231
        # 1        000596.SZ       000729.SZ  4643.178428  20221231
        # 2        000596.SZ       000738.SZ   531.885128  20221231
        # 3        000596.SZ       000786.SZ   526.061043  20221231
        # 4        000596.SZ       000807.SZ    85.074528  20221231
        if 'lnK1' not in df.columns:
            df = df[df.stockcode.apply(lambda x: x[:6]) < df.stockcode_right.apply(lambda x: x[:6])]
            if 'lnK' not in df.columns:
                df['lnK'] = df.K.apply(np.log)  # 取对数
            df['lnK1'] = df['lnK'].sub(df['lnK'].min()).div(df['lnK'].max() - df['lnK'].min())
            df.dropna(inplace=True)
            df.to_pickle(src); print('update:', src)
        similarities = df 
        
        # 两两各自的风格类别
        if kind == 1:
            fv_per = mv.loc[mv.loc[:period].index[-1]].reindex(similarities['stockcode'].unique())\
                .rank(pct=True).mul(group_num).rsub(group_num).fillna(-1).astype(int).replace(-1, None)
        elif kind == 2:
            fv_per = mv.loc[mv.loc[:period].index[-1]]\
                .rank(pct=True).mul(group_num).rsub(group_num).fillna(-1).astype(int).replace(-1, None)\
                .reindex(similarities['stockcode'].unique())
        similarities[fname] = fv_per.reindex(similarities['stockcode']).values
        similarities[fname + '_right'] = fv_per.reindex(similarities['stockcode_right']).values
        # 风格类别相同
        similarities[f'iden_{fname}'] = similarities[fname].eq(similarities[f'{fname}_right']).astype(int)

        # 相同风格分组下的关联度指标统计量
        res = similarities.query(f"iden_{fname} == 1").groupby(fname)['lnK1'].describe()
        res = res.reset_index()
        res[fname] = res[fname].astype(int)
        res['count'] = res['count'].astype(int)
        res['period'] = period
        res = res.set_index(['period', fname])
        print(res)
        return res
    
    if (tab_path is None) or (not os.path.exists(tab_path)) or force_update:
        res = {}
        for pers in ((y*10000+630, y*10000+1231) 
                     for y in range(2013, 2023)):
            res[pers[0]] = per_comp(pers[0])
            res[pers[1]] = per_comp(pers[1])
        tab = pd.concat(res.values())
        if tab_path is not None:
            tab.to_excel(tab_path)
            print(tab_path)
    else:
        tab = pd.read_excel(tab_path, index_col=[0, 1])

    col2name =  {0: f'大市值', group_num - 1: f'小市值'}
    col2name.update({gn: f'第 {gn+1} 组' for gn in range(1, group_num-1)})
    g = tab['mean'].unstack().rename(columns=col2name)
    g.columns.name = None
    g.index = g.index.astype(str).rename('报告期')
    plt.figure()
    for i in range(group_num):
        if i < g.shape[1]:
            plt.plot(g.index, g.iloc[:, i], linewidth=3, linestyle=('-', '--')[i%2],
                     color='k', alpha=1 - i * (1 / group_num), label=g.columns[i])
    plt.xticks(rotation=45, ha='right', va='top')
    plt.xlabel('报告期')
    plt.ylim(0, 1.0)
    plt.ylabel('平均关联度')
    plt.legend(frameon=False)
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


def man_lnK1_barra_comp(fname='MV', kw='市值', group_num=5, tab_path=None, fig_path=None, force_update=False):
    """不同风格分组 值对比"""

    fval = pd.read_hdf(f"data/QUANTILE_{fname}.h5", key='data')
    # >>> fval.head()
    # date      instrument
    # 20100104  000001.SZ     0
    #           000002.SZ     0
    #           000004.SZ     9
    #           000005.SZ     2
    #           000006.SZ     2
    # Name: QUANTILE_MV, dtype: int64

    # period = 20221231
    def per_comp(period):
        """比较一个报告期"""
        src = f"cache/sim_Kab.{period}.pkl"
        if not os.path.exists(src):
            print(src)
            return None
        df = pd.read_pickle(src)
        # >>> df
        #          stockcode stockcode_right            K    period
        # 0        000596.SZ       000630.SZ    51.891086  20221231
        # 1        000596.SZ       000729.SZ  4643.178428  20221231
        # 2        000596.SZ       000738.SZ   531.885128  20221231
        # 3        000596.SZ       000786.SZ   526.061043  20221231
        # 4        000596.SZ       000807.SZ    85.074528  20221231
        if 'lnK1' not in df.columns:
            df = df[df.stockcode.apply(lambda x: x[:6]) < df.stockcode_right.apply(lambda x: x[:6])]
            if 'lnK' not in df.columns:
                df['lnK'] = df.K.apply(np.log)  # 取对数
            df['lnK1'] = df['lnK'].sub(df['lnK'].min()).div(df['lnK'].max() - df['lnK'].min())
            df.dropna(inplace=True)
            df.to_pickle(src); print('update:', src)
        similarities = df 
        
        # 两两各自的风格类别
        fv_per = fval.loc[fval.loc[:period].index[-1][0]] // 2
        similarities[fname] = fv_per.reindex(similarities['stockcode']).values
        similarities[fname + '_right'] = fv_per.reindex(similarities['stockcode_right']).values
        # 风格类别相同
        similarities[f'iden_{fname}'] = similarities[fname].eq(similarities[f'{fname}_right']).astype(int)
        
        # 相同风格分组下的关联度指标统计量
        res = similarities.query(f"iden_{fname} == 1").groupby(fname)['lnK1'].describe()
        res = res.reset_index()
        res[fname] = res[fname].astype(int)
        res['count'] = res['count'].astype(int)
        res['period'] = period
        res = res.set_index(['period', fname])
        print(res)
        return res
    
    if (tab_path is None) or (not os.path.exists(tab_path)) or force_update:
        res = {}
        for pers in ((y*10000+630, y*10000+1231) 
                     for y in range(2013, 2023)):
            res[pers[0]] = per_comp(pers[0])
            res[pers[1]] = per_comp(pers[1])
        tab = pd.concat(res.values())
        if tab_path is not None:
            tab.to_excel(tab_path)
            print(tab_path)
    else:
        tab = pd.read_excel(tab_path, index_col=[0, 1])


    g = tab['mean'].unstack().rename(columns={0: f'高{kw}', 1: '第二组',
                                              2: '第三组', 3: '第四组',
                                              4: f'低{kw}'})
    g.columns.name = None
    g.index = g.index.astype(str).rename('报告期')

    plt.figure()
    for i in range(g.shape[1]):
        plt.plot(g.index, g.iloc[:, i], linewidth=3, linestyle=('-', '--')[i%2], color='k', alpha=1 - i * (1 / group_num), label=g.columns[i])
    plt.xticks(rotation=45, ha='right', va='top')
    plt.xlabel('报告期')
    plt.ylim(0, 1.0)
    plt.ylabel('平均关联度')
    plt.legend(frameon=False)
    plt.grid(axis='y')
    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path, transparent=True)
        print(fig_path)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':

    # for fname in ['AMV', 'EPS', 'PB', 'PE', 'LEVG', 'TVR', 'RET', 'STD', 'MV']:
    #     man_lnK1_barra_comp(
    #         fname=fname,
    #         kw=fname,
    #         tab_path=f'results/tab_lnK1_comp_{fname}.xlsx',
    #         fig_path=f'results/fig_lnK1_comp_{fname}.png',
    #         force_update=False
    #     )
    # man_lnK1_indus_comp(
    #     tab_path='results/tab_lnK1_comp.xlsx',
    #     fig_path='results/fig_lnK1_comp.png',
    #     force_update=False
    # )
    # man_sim_mv_comp(
    #     tab_path='results/tab_lnK1_comp_gamv.xlsx',
    #     fig_path='results/fig_lnK1_comp_gamv.png',
    #     kind=1,
    #     amv=True,
    #     force_update=False
    # )
    # man_sim_mv_comp(
    #     tab_path='results/tab_lnK1_comp_amv.xlsx',
    #     fig_path='results/fig_lnK1_comp_amv.png',
    #     kind=2,
    #     amv=True,
    #     force_update=False
    # )
    
    man_ret_r2(
        tab_path='results/tab_lnK1r2_comp_g5_s100000.xlsx',
        fig_path='results/tab_lnK1r2_comp_g5_s100000.png',
        group_num=5,
        sample_n=100000,
        force_update=False
    )

    # w2v_sim_indus_comp(
    #     tab_path='results/tab_sim_comp.xlsx',
    #     fig_path='results/fig_sim_comp.png',
    #     force_update=False
    # )
    # w2v_sim_mv_comp(
    #     tab_path='results/tab_sim_comp_amv.xlsx',
    #     fig_path='results/fig_sim_comp_amv.png',
    #     kind=2,
    #     amv=True,
    #     force_update=False
    # )
    # w2v_sim_mv_comp(
    #     tab_path='results/tab_sim_comp_gamv.xlsx',
    #     fig_path='results/fig_sim_comp_gamv.png',
    #     kind=1,
    #     amv=True,
    #     force_update=False
    # )
    
    w2v_ret_r2(
        tab_path='results/tab_r2_comp_g5_s100000.xlsx',
        fig_path='results/tab_r2_comp_g5_s100000.png',
        group_num=5,
        sample_n=100000,
        force_update=False
    )
    
    # for fname in ['AMV', 'EPS', 'PB', 'PE', 'LEVG', 'TVR', 'RET', 'STD', 'MV']:
    #     w2v_sim_barra_comp(
    #         fname=fname,
    #         kw=fname,
    #         tab_path=f'results/tab_sim_comp_{fname}.xlsx',
    #         fig_path=f'results/fig_sim_comp_{fname}.png',
    #         force_update=False
    #     )
