"""
I. sentence preparation: (period, fundcode)

- sentences: rank, $\rho_{i a}$, of asset $a$ in investor $i$'s portfolio

Steps:
------

1. Load investor position for each period: (period, fundcode, stockcode, amount)
2. Skip special period
3. Drop small stocks
4. Drop non-universe stocks (ST)
5. Drop funds not focus on ashare (amount in A share less than 20 percents)
6. Drop minor stocks (holded by less than K=10 funds)
7. Drop minor investors (holded less than K=10 major stocks)
8. Build sentences
    - holdings
    - active weights
    - rebalancing

"""
import os
from pathlib import Path
import pandas as pd


_PATH = '/home/winston/Thesis/'
os.chdir(_PATH)


def data_fund(tgt=None, force_update=False) -> pd.DataFrame:
    """
    1. Load investor position for each period: (period, fundcode, stockcode, amount)

    Return
    ------ 
              period   fundcode  stockcode tradingdate        amount  shrpct
    299051  20130630  000001.OF  000001.SZ    20130826  6.380800e+07    0.72
    299052  20130630  000001.OF  000002.SZ    20130719  1.918581e+08    2.16
    299053  20130630  000001.OF  000061.SZ    20130826  5.660000e+07    0.64
    299054  20130630  000001.OF  000069.SZ    20130826  2.098970e+07    0.24
    299055  20130630  000001.OF  000100.SZ    20130826  1.592655e+07    0.18
    
    """
    kw = '中国共同基金投资组合——持股明细'
    src = "./data/%s.pkl" % kw
    if tgt is None:
        tgt = './cache/data_fund.pkl'
    if os.path.exists(tgt) and (not force_update):
        return pd.read_pickle(tgt)
    else:
        df = pd.read_pickle(src)[kw]  # Load raw dataframe
        df = df.rename(columns={
            'fundCode': 'fundcode',
            'tradeDate': 'tradingdate', 
            '报告期': 'period',
            'stockCode': 'stockcode',
            '持有股票市值(元)': 'amount',
            '持有股票市值占资产净值比例(%)': 'shrpct',
        })
        # Drop duplicated record (only once) - `001980.OF  20221231  600188.SH`
        df = df.groupby(['period', 'fundcode', 'stockcode']).last().reset_index()
        # Save local cache
        df.to_pickle(tgt)
        return df


def data_fundinfo() -> pd.DataFrame:
    """
    Return
    ------
           fundCode   fundName        成立日期  到期日期                             业绩比较基准 基金类型 基金风格    开放情况          wind基金类型        进入日期        离开日期 wind基金名称  基金名称不含AC
    5076  000001.OF       华夏成长  20011218.0   NaN                                NaN  混合型  成长型  契约型开放式  2001010201000000  20011218.0  20210919.0  偏股混合型基金      华夏成长
    5075  000001.OF       华夏成长  20011218.0   NaN                                NaN  混合型  成长型  契约型开放式  2001010204000000  20210920.0         NaN  灵活配置型基金      华夏成长
    7560  000006.OF  西部利得量化成长A  20190319.0   NaN  中证500指数收益率*75%+同期银行活期存款利率(税后)*25%  混合型  混合型  契约型开放式  2001010201000000  20190319.0         NaN  偏股混合型基金  西部利得量化成长
    2986  000011.OF    华夏大盘精选A  20040811.0   NaN        富时中国A200指数*80%+富时中国国债指数*20%  混合型  增值型  契约型开放式  2001010204000000  20210920.0         NaN  灵活配置型基金    华夏大盘精选
    2987  000011.OF    华夏大盘精选A  20040811.0   NaN        富时中国A200指数*80%+富时中国国债指数*20%  混合型  增值型  契约型开放式  2001010201000000  20040811.0  20210919.0  偏股混合型基金    华夏大盘精选

    """
    kw = '20081231_20230817_权益型公募基金信息'
    src = './data/%s.xlsx' % kw
    fundinfo = pd.read_excel(src, index_col=0)
    return fundinfo


def data_universe(kw='tradableBar5') -> pd.DataFrame:
    """
    Get universe label

    Return
    ------
    stockcode    000001.SZ  000002.SZ  000004.SZ  000005.SZ  000006.SZ
    tradingdate                                                       
    2010-01-04         1.0        1.0        NaN        1.0        1.0
    2010-01-05         1.0        1.0        NaN        1.0        1.0
    2010-01-06         1.0        1.0        NaN        1.0        1.0
    2010-01-07         1.0        1.0        NaN        1.0        1.0
    2010-01-08         1.0        1.0        NaN        1.0        1.0

    """
    src = Path(f'./data/{kw}.h5')
    df = pd.read_hdf(src, key=kw)
    df = df['valid'].unstack()
    df.index = pd.to_datetime(df.index)
    df.index.name = 'tradingdate'
    df.columns.name = 'stockcode'
    return df


def data_asset_mv(kw="总市值") -> pd.DataFrame:
    """
    Get asset market share

    Return
    ------
    stockcode       000001.SZ     000002.SZ 000003.SZ   000004.SZ    000005.SZ
    tradingdate                                                               
    2008-12-31   2.634237e+06  6.071939e+06       NaN  25029.4383  180916.6354
    2009-01-05   2.703852e+06  6.307285e+06       NaN  25943.4347  186752.6559
    2009-01-06   2.868144e+06  6.495562e+06       NaN  26716.8162  192588.6764
    2009-01-07   2.781821e+06  6.457907e+06       NaN  28052.6570  190400.1687
    2009-01-08   2.673222e+06  6.495562e+06       NaN  29458.8052  185293.6507 

    """
    src = "data/ashareMarketData/日行情/%s.h5" % kw
    mv = pd.read_hdf(src)
    mv.index = pd.to_datetime(mv.index, format='%Y%m%d').rename('tradingdate')
    mv.columns.name = 'stockcode'
    return mv


def data_mv(kw="流通市值") -> pd.DataFrame:
    """
    Get free market share

    Return
    ------
    stockcode       000001.SZ     000002.SZ 000003.SZ   000004.SZ    000005.SZ
    tradingdate                                                               
    2008-12-31   2.634237e+06  6.071939e+06       NaN  25029.4383  180916.6354
    2009-01-05   2.703852e+06  6.307285e+06       NaN  25943.4347  186752.6559
    2009-01-06   2.868144e+06  6.495562e+06       NaN  26716.8162  192588.6764
    2009-01-07   2.781821e+06  6.457907e+06       NaN  28052.6570  190400.1687
    2009-01-08   2.673222e+06  6.495562e+06       NaN  29458.8052  185293.6507 

    """
    src = "data/ashareMarketData/日行情/%s.h5" % kw
    mv = pd.read_hdf(src)
    mv.index = pd.to_datetime(mv.index, format='%Y%m%d').rename('tradingdate')
    mv.columns.name = 'stockcode'
    return mv


def only_universe(df, univ: pd.DataFrame=None) -> pd.DataFrame:
    """Drop non-universe stocks"""
    dt_eop = pd.to_datetime(df.period.iloc[0], format='%Y%m%d')
    valid_eop = univ.loc[univ.loc[:dt_eop].index[-1]]
    stk_valid = valid_eop[valid_eop == 1.0].index.to_list()
    return df[df.stockcode.isin(stk_valid)]


def no_small_stock(df, mv, bar=0.2) -> pd.DataFrame:
    """Drop rows where stockcode is `bar` smallest stocks."""
    # Date - end of period
    dt_eop = pd.to_datetime(df.period.value_counts().index[0], format='%Y%m%d')
    # MV on / befor the end of period
    mv_eop: pd.Series = mv.loc[:dt_eop].iloc[-1]
    # Big MV stocks list
    stk_big = mv_eop[mv_eop.rank(ascending=True, pct=True) >= bar].index.to_list()
    return df[df.stockcode.isin(stk_big)]


def sift_main_funds(df: pd.DataFrame, centre='fundcode', val='shrpct', bar=0.2) -> pd.DataFrame:
    """Drop funds that invest less than `bar`% of wealth in A-Share stocks"""
    tmp = df.groupby(centre)[val].sum()
    tmp = tmp[tmp >= (bar * 100)]
    mask = df[centre].isin(tmp.index)
    return df[mask]


def sift_min_obs(df: pd.DataFrame, col: str, k=10) -> pd.DataFrame:
    """Drop if value in column `col` appears less than `k` times"""
    tmp = df[col].value_counts()
    tmp = tmp[tmp >= k]
    mask = df[col].isin(tmp.index)
    return df[mask]


def sentences_from(df, per='period', centre='fundcode', val='amount', td='tradingdate', name='stockcode', asc=False) -> pd.DataFrame:
    """Build sentences of each [`per`, `centre`] with a list of `name` ordered by ranked value `val`"""
    kw = 'sentence'
    tmp = df.groupby(centre).apply(lambda s: s.sort_values(val, ascending=asc)[name].to_list())
    tmp = tmp.rename(kw)
    tmp = tmp.reset_index()
    tmp[per] = df[per].value_counts().index[0]
    tmp[td] = tmp[centre].apply(lambda x: df.query(f"{centre} == '{x}'")[td].max())
    return tmp[[td, per, centre, kw]]


def dump_dataframe(sr: pd.Series, file: str):
    """Save in local file"""
    file = Path(file)
    if file.suffix == '.pkl':
        sr.to_pickle(file)
    elif file.suffix == '.txt':
        with open(file, 'w', encoding='utf-8') as f:
            for ind in sr.index:
                f.write(f'{ind}\n')
                f.write(f'{sr.loc[ind]}\n')
    else:
        raise Exception('Invalid sentence cache dir', file)


def process_data(K=10, force_update=False):
    """Prepare sentences."""

    # Output path
    tgt0 = f'cache/sentences.{K}.pkl'
    tgt1 = f'cache/holdamt.{K}.pkl'
    if (os.path.exists(tgt0) and os.path.exists(tgt1)) and not force_update:
        return tgt0, tgt1

    # 1. Load investor position for each period: (period, fundcode, stockcode, amount)
    df = data_fund().query("20130101 <= period <= 20221231")

    # Fund information
    finfo = data_fundinfo()
    
    # 2. Skip special periods  只关注报告期为 0630 和 1231 的基金持仓披露
    mask = (df.period % 10000).isin([630,1231])
    # mask = (df.period % 10000).isin([331, 630, 930, 1231])
    df = df[mask]  # Caution: potentially insufficient disclosure for 20230331, 20230630

    # Universe  可交易股票
    univ0: pd.DataFrame = data_universe(kw='tradableBar5')

    # Market value
    mv2d: pd.DataFrame = data_mv()

    # For each period, bulid sentences
    sentences_k = []
    holdamt_k = []
    for per in df.period.unique():  # 每个季度末的基金持仓情况

        print(f"-------\n{per}")
        df1 = df[df.period == per]

        # Drop funds not focusing on ashare  去除持仓A股金额占比少于20%的基金
        df1 = sift_main_funds(df1, centre='fundcode', val='shrpct', bar=0.2)
        
        # Drop small stocks  去除每个截面市值最小20%的股票
        df1 = no_small_stock(df1, mv=mv2d, bar=0.2)
  
        # Drop non-universe stocks  去除 停牌/ST/ST*/新上市 的股票  TODO: 额外去除了季末涨停
        df1 = only_universe(df1, univ=univ0)

        # Drop minor stocks  去除每期被持有的基金数量不足 K=10 的股票
        df1 = sift_min_obs(df1, col='stockcode', k=K)
        print("# tokens:\t%d" % df1.stockcode.unique().__len__())
        
        # Drop minor investors  去除每期持有股票数量不足 K=10 的基金
        df1 = sift_min_obs(df1, col='fundcode', k=K)
        print("# sentences:\t%d" % df1.fundcode.unique().__len__())

        # Build sentences
        sentence = sentences_from(df1)

        sentences_k.append(sentence)
        holdamt_k.append(df1)

    sentences_k = pd.concat(sentences_k)[
        ['tradingdate', 'period', 'fundcode', 'sentence']
    ].sort_values(['tradingdate', 'period', 'fundcode']).reset_index(drop=True)
    holdamt_k = pd.concat(holdamt_k, axis=0)[
        ['tradingdate', 'period', 'fundcode', 'stockcode', 'amount', 'shrpct']
    ].sort_values(['tradingdate', 'period', 'fundcode', 'amount']).reset_index(drop=True)
    
    # Save in local cache path
    dump_dataframe(sentences_k, file=tgt0)
    dump_dataframe(holdamt_k, file=tgt1)

    return tgt0, tgt1


if __name__ == '__main__':
    print('sentences saved in:\t%s' % process_data())
