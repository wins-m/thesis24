"""

Prepare tokens.

Steps:
------

1. Load investor position for each period: (period, fundcode, stockcode, amount)
2. Skip special period
        # Drop funds not focus on ashare
3. Drop non-universe stocks
4. Drop small stocks
5. Drop minor stocks
6. Drop minor investors

"""
import os
from pathlib import Path
import pandas as pd


_PATH = '/home/winston/Thesis/'
os.chdir(_PATH)


def data_fund() -> pd.DataFrame:
    """1. Load investor position for each period: (period, fundcode, stockcode, amount)"""
    tgt = './cache/data_fund.pkl'
    if os.path.exists(tgt):
        return pd.read_pickle(tgt)
    else:
        # Load raw dataframe
        kw = '中国共同基金投资组合——持股明细'
        src = "./data/%s.pkl" % kw
        df = pd.read_pickle(src)[kw]
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


def data_fundinfo():
    kw = '20081231_20230817_权益型公募基金信息'
    src = './data/%s.xlsx' % kw
    fundinfo = pd.read_excel(src, index_col=0)
    return fundinfo


def data_universe() -> pd.Series:
    """Get universe label"""
    kw = 'tradableBar5'
    src = Path(f'./data/{kw}.h5')
    df = pd.read_hdf(src, key=kw)
    df = df.reset_index()
    df = df.rename(columns={'date': 'tradingdate', 'instrument': 'stockcode', 'valid': 'valid'})
    df['tradingdate'] = pd.to_datetime(df['tradingdate'])
    df = df.set_index(['tradingdate', 'stockcode']).sort_index()
    return df


def data_mv():
    """Get free market share"""
    kw = "流通市值"
    src = "data/ashareMarketData/日行情/%s.h5" % kw
    mv = pd.read_hdf(src)
    mv.index = pd.to_datetime(mv.index, format='%Y%m%d').rename('tradingdate')
    mv.columns.name = 'stockcode'
    return mv


def only_universe(df, univ=None, per=None) -> pd.DataFrame:
    """TODO Drop non-universe stocks"""
    return df


def no_small_stock(df, mv, bar=0.2) -> pd.DataFrame:
    """Drop rows where stockcode is `bar` smallest stocks."""
    # Date - end of period
    dt_eop = pd.to_datetime(df.period.value_counts().index[0], format='%Y%m%d')
    # MV on / befor the end of period
    mv_eop: pd.Series = mv.loc[:dt_eop].iloc[-1]
    # Big MV stocks list
    stk_big = mv_eop[mv_eop.rank(ascending=True, pct=True) >= bar].index.to_list()
    return df[df.stockcode.isin(stk_big)]


def sift_main_funds(df, centre='fundcode', val='shrpct', bar=20):
    """Drop funds that invest less than `bar`% of wealth in A-Share stocks"""
    tmp = df.groupby(centre)[val].sum()
    tmp = tmp[tmp >= bar]
    mask = df[centre].isin(tmp.index)
    return df[mask]


def sift_min_obs(df: pd.DataFrame, col: str, k=10) -> pd.DataFrame:
    """Drop if value in column `col` appears less than `k` times"""
    tmp = df[col].value_counts()
    tmp = tmp[tmp >= k]
    mask = df[col].isin(tmp.index)
    return df[mask]


def tokens_from(df, per='period', centre='fundcode', val='amount', name='stockcode', asc=False):
    """Build tokens of each [`per`, `centre`] with a list of `name` ordered by ranked value `val`"""
    kw = 'tokens'
    tmp = df.groupby(centre).apply(lambda s: s.sort_values(val, ascending=asc)[name].to_list())
    tmp = tmp.rename(kw)
    tmp = tmp.reset_index()
    tmp[per] = df[per].value_counts().index[0]
    return tmp.set_index([per, centre])[kw]


def dump_tokens(sr, file):
    """Save tokens in local file"""
    file = Path(file)
    if file.suffix == '.pkl':
        sr.to_pickle(file)
    elif file.suffixe == '.txt':
        with open(file, 'w', encoding='utf-8') as f:
            for ind in sr.index:
                f.write(f'{ind}\n')
                f.write(f'{sr.loc[ind]}\n')
    else:
        raise Exception('Invalid token cache dir', file)
    print('Tokens saved in:\n\t', file)


def process_data(tgt='./cache/tokens.pkl', K=10):
    """Prepare tokens."""

    # 1. Load investor position for each period: (period, fundcode, stockcode, amount)
    df = data_fund()
    df = df[df.period >= 20130101]
    df = df[df.period < 20230101]
    
    # 2. Skip special periods
    mask = (df.period % 10000).isin([630,1231])
    # mask = (df.period % 10000).isin([331, 630, 930, 1231])
    df = df[mask]  # Caution: potentially insufficient disclosure for 20230331, 20230630

    # Fund information
    finfo = data_fundinfo()

    # Universe
    univ0 = data_universe()

    # Market value
    mv2d = data_mv()

    # For each period, bulid tokens
    tokens = []
    for per in df.period.unique():
        print("-------\n", per)
        df1 = df[df.period == per]

        # Drop funds not focus on ashare
        df1 = sift_main_funds(df1, centre='fundcode', val='shrpct', bar=20)

        # 3. Drop non-universe stocks
        df1 = only_universe(df1, univ=univ0)

        # 4. Drop small stocks
        df1 = no_small_stock(df1, mv=mv2d)
  
        # 5. Drop minor stocks
        df1 = sift_min_obs(df1, col='stockcode', k=K)
        print("# vocabulary:\t%d" % df1.stockcode.unique().__len__())
        
        # 6. Drop minor investors
        df1 = sift_min_obs(df1, col='fundcode', k=K)
        print("# tokens:\t%d" % df1.fundcode.unique().__len__())

        # 7. Build tokens
        tokens.append(tokens_from(df1))

    tokens = pd.concat(tokens).sort_index()
    
    # Save in local cache path
    dump_tokens(tokens, file=tgt)


if __name__ == '__main__':
    process_data()
