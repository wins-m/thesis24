"""
Always modify rawdata I/O for different machines.
https://github.com/hobinkwak/Stock2Vec-Inverse-Volatility/blob/main/utils/utils.py

"""
import pandas as pd
from functools import reduce


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
    univ0 = get_universe()
    price_df = get_price_df(univ=univ0)
    rebal_dates = get_rebal_dates(fv2d=price_df, start_year='2009')
    print(price_df.loc[rebal_dates].count(1))
    pass


if __name__ == '__main__':
    debug()