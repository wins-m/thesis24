"""
Always modify rawdata I/O for different machines.
https://github.com/hobinkwak/Stock2Vec-Inverse-Volatility/blob/main/utils/utils.py

"""
import pandas as pd
from functools import reduce


def data_adjopen() -> pd.DataFrame:
    """
    stockcode    000001.SZ  000002.SZ 000003.SZ  000004.SZ  000005.SZ
    tradingdate                                                      
    20081231        342.90     716.86       NaN      14.22      23.73
    20090105        343.62     725.68       NaN      14.63      23.08
    20090106        351.87     736.71       NaN      14.96      23.82
    20090107        366.24     764.28       NaN      15.32      24.47
    20090108        350.08     744.43       NaN      16.21      24.00
    """
    adjopen = pd.read_hdf('data/复权开盘价.h5', key='data')
    adjopen.index.name = 'tradingdate'
    adjopen.columns.name = 'stockcode'
    return adjopen


def get_winsorize_sr(sr: pd.Series, nsigma=3) -> pd.Series:
    """对series缩尾"""
    sr1 = sr.copy()
    md = sr1.median()
    mad = 1.483 * sr1.sub(md).abs().median()
    up = sr1.apply(lambda k: k > md + mad * (0 + nsigma))
    down = sr1.apply(lambda k: k < md - mad * (0 + nsigma))
    sr1[up] = sr1[up].rank(pct=True).multiply(mad * 0.5).add(md + mad * (0 + nsigma))
    sr1[down] = sr1[down].rank(pct=True).multiply(mad * 0.5).add(md - mad * (0 + nsigma))
    return sr1


def get_last_period(stime: int, freq="HY") -> int:
    """freq 频率 返回上一个报告期"""
    y, d = stime // 10000, stime % 10000
    if freq == 'HY':
        if d == 630:
            y -= 1
            d = 1231
        else:
            d = 630
        return y * 10000 + d


def get_next_period(stime: int, freq="HY") -> int:
    """freq 频率 返回下一个报告期"""
    y, d = stime // 10000, stime % 10000
    if freq == 'HY':
        if d == 630:
            d = 1231
        else:
            y += 1
            d = 630
        return y * 10000 + d


def get_industry(stime=20130101, etime=20231231, lvl=1) -> pd.Series:
    src = 'data/industries_class_citics.1.h5'
    df = pd.read_hdf(src, key='data').query(f'{stime} <= date <= {etime}')
    return df[f"CODELV{lvl}"]


def get_universe(bd='2009-01-01', ed='2023-06-30', src=None) -> pd.DataFrame:
    if src is None:
        src = "data/中证1000.h5"
    df = pd.read_hdf(src)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df.index.name = 'tradingdate'
    df.columns.name = 'stockcode'
    df = df.loc[bd: ed].dropna(axis=1, how='all')
    return df


def get_price_df(univ: pd.DataFrame, src=None):
    if src is None:
        src = "data/复权开盘价.h5"
    # Daily price
    df = pd.read_hdf(src)
    df.index = pd.to_datetime(df.index, format='%Y%m%d')
    df.index.name = 'tradingdate'
    df.columns.name = 'stockcode'
    # Keep univ only
    df = df.reindex_like(univ)
    df = df * univ
    return df


def get_rebal_dates(fv2d, start_year='2009'):
    """Rebalance on last tradedate of each month."""
    df = pd.DataFrame(index=fv2d.index)
    df['month'] = fv2d.index.month
    df['year'] = fv2d.index.year
    df = df.drop_duplicates(subset=['year', 'month'], keep='last').loc[start_year:].index
    return df 


def compute_portfolio_cum_rtn(price_df, weights):
    cum_rtn = 1
    individual_port_val_df_list = []

    prev_end_day = weights.index[0]
    for end_day in weights.index[1:]:
        sub_price_df = price_df.loc[prev_end_day:end_day]
        sub_asset_flow_df = sub_price_df / sub_price_df.iloc[0]

        weight_series = weights.loc[prev_end_day]
        indi_port_cum_rtn_series = (sub_asset_flow_df * weight_series) * cum_rtn

        individual_port_val_df_list.append(indi_port_cum_rtn_series)

        total_port_cum_rtn_series = indi_port_cum_rtn_series.sum(axis=1)
        cum_rtn = total_port_cum_rtn_series.iloc[-1]

        prev_end_day = end_day

    individual_port_val_df = reduce(lambda x, y: pd.concat([x, y.iloc[1:]]), individual_port_val_df_list)
    return individual_port_val_df


def debug():
    indus = get_industry()
    univ0 = get_universe()
    price_df = get_price_df(univ=univ0)
    rebal_dates = get_rebal_dates(fv2d=price_df, start_year='2009')
    print(price_df.loc[rebal_dates].count(1))
    pass


if __name__ == '__main__':
    debug()