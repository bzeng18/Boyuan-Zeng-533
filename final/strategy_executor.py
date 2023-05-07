import pandas as pd

import numpy as np
import os
import refinitiv.dataplatform.eikon as ek
import refinitiv.data as rd
import math
import sys
#####################################################

def render_blotter():
    ek.set_app_key(os.getenv('EIKON_APIBZ'))
    ivv_prc = pd.read_csv('prices.csv')

    ivv_prc['Date'] = pd.to_datetime(ivv_prc['Date']).dt.date
    ivv_prc.drop(columns='Instrument', inplace=True)


    ##### Get the next business day from Refinitiv!!!!!!!
    rd.open_session()

    next_business_day = rd.dates_and_calendars.add_periods(
        start_date= ivv_prc['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )

    rd.close_session()
    ######################################################
    # Parameters:
    alpha1 = -0.01
    n1 = 3
    alpha2 = 0.01
    n2 = 5

    # submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, ivv_prc.shape[0]),
        "date": list(pd.to_datetime(ivv_prc["Date"].iloc[1:]).dt.date),
        "asset": "IVV",
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(
            ivv_prc['Close Price'].iloc[:-1] * (1 + alpha1),
            2
        ),
        'status': 'SUBMITTED'
    })

    # if the lowest traded price is still higher than the price you bid, then the
    # order never filled and was cancelled.
    with np.errstate(invalid='ignore'):
        cancelled_entry_orders = submitted_entry_orders[
            np.greater(
                ivv_prc['Low Price'].iloc[1:][::-1].rolling(n1).min()[
                ::-1].to_numpy(),
                submitted_entry_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_entry_orders.reset_index(drop=True, inplace=True)
    cancelled_entry_orders['status'] = 'CANCELLED'
    cancelled_entry_orders['date'] = pd.DataFrame(
        {'cancel_date': submitted_entry_orders['date'].iloc[(n1-1):].to_numpy()},
        index=submitted_entry_orders['date'].iloc[:(1-n1)].to_numpy()
    ).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()

    filled_entry_orders = submitted_entry_orders[
        submitted_entry_orders['trade_id'].isin(
            list(
                set(submitted_entry_orders['trade_id']) - set(
                    cancelled_entry_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_entry_orders.reset_index(drop=True, inplace=True)
    filled_entry_orders['status'] = 'FILLED'
    for i in range(0, len(filled_entry_orders)):

        idx1 = np.flatnonzero(
            ivv_prc['Date'] == filled_entry_orders['date'].iloc[i]
        )[0]

        ivv_slice = ivv_prc.iloc[idx1:(idx1+n1)]['Low Price']

        fill_inds = ivv_slice <= filled_entry_orders['price'].iloc[i]

        if (len(fill_inds) < n1) & (not any(fill_inds)):
            filled_entry_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_entry_orders.at[i, 'date'] = ivv_prc['Date'].iloc[
                fill_inds.idxmax()
            ]

    live_entry_orders = pd.DataFrame({
        "trade_id": ivv_prc.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": "IVV",
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(ivv_prc['Close Price'].iloc[-1] * (1 + alpha1), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    if any(filled_entry_orders['status'] =='LIVE'):
        live_entry_orders = pd.concat([
            filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
            live_entry_orders
        ])
        # "today" is the next business day after the last closing price
        live_entry_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]

    entry_orders = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            live_entry_orders
        ]
    ).sort_values(["date", 'trade_id'])


    # for every filled entry order, there must exist a submitted exit order:
    submitted_exit_orders = filled_entry_orders.copy()
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = 'SELL'
    submitted_exit_orders['price'] = submitted_exit_orders['price'] * (1 + alpha2)
    submitted_exit_orders['status'] = 'SUBMITTED'

    # Figure out what happened to each exit order we submitted
    exit_order_fates = submitted_exit_orders.copy()
    exit_mkt_orders = pd.DataFrame(columns=exit_order_fates.columns)

    for index, exit_order in submitted_exit_orders.iterrows():

        # was it filled the day it was submitted?
        if float(
                ivv_prc.loc[ivv_prc['Date'] == exit_order['date'], 'Close Price']
        ) >= exit_order['price']:
            exit_order_fates.at[index, 'status'] = 'FILLED'
            continue

        window_prices = ivv_prc[ivv_prc['Date'] > exit_order['date']].head(n2)

        # was it submitted on the very last day for which we have data?
        if window_prices.size == 0:
            exit_order_fates.at[index, 'date'] = pd.to_datetime(
                next_business_day).date()

            exit_order_fates.at[index, 'status'] = 'LIVE'
            continue

        filled_ind, *asdf = np.where(
            window_prices['High Price'] >= exit_order['price']
        )

        if filled_ind.size == 0:

            if window_prices.shape[0] < n2:
                exit_order_fates.at[index, 'date'] = pd.to_datetime(
                    next_business_day).date()

                exit_order_fates.at[index, 'status'] = 'LIVE'
                continue

            exit_order_fates.at[index, 'date'] = window_prices['Date'].iloc[
                window_prices.shape[0] - 1
                ]
            exit_order_fates.at[index, 'status'] = 'CANCELLED'
            exit_mkt_orders = pd.concat([
                exit_mkt_orders,
                pd.DataFrame({
                    'trade_id': exit_order['trade_id'],
                    'date': window_prices['Date'].tail(1),
                    'asset': exit_order['asset'],
                    'trip': exit_order['trip'],
                    'action': exit_order['action'],
                    'type': "MKT",
                    'price': window_prices['Close Price'].tail(1),
                    'status': 'FILLED'
                })
            ])
            continue

        exit_order_fates.at[index, 'date'] = window_prices['Date'].iloc[
                min(filled_ind)
        ]
        exit_order_fates.at[index, 'status'] = 'FILLED'


    blotter = pd.concat(
        [entry_orders, submitted_exit_orders, exit_order_fates, exit_mkt_orders]
    ).sort_values(['trade_id', "date", 'trip']).reset_index(drop=True)

    return blotter


def blotter_to_ledger(blotter):
    ledger = pd.DataFrame(columns=['trade_id', 'asset', 'dt_enter', 'dt_exit', 'success', 'n', 'rtn'])

    # Calculate the length of the trade, use numpy built-in function count business day, but need to convert the date to <M8[D] type first.
    def calculate_n(start_date, end_date):
        start_day = pd.to_datetime(orders[start_date]['date'].values[0]).to_numpy().astype('<M8[D]')
        end_day = pd.to_datetime(orders[end_date]['date'].values[0]).to_numpy().astype('<M8[D]')
        return np.busday_count(start_day, end_day) + 1

    def calculate_rtn(start_price, end_price, n):
        return math.log(end_price / start_price) / n

    for trade_id in blotter['trade_id'].unique():
        orders = blotter[blotter['trade_id'] == trade_id]
        is_submitted_entry_order = ((orders['trip'] == 'ENTER') & (orders['status'] == 'SUBMITTED'))
        is_filled_entry_order = ((orders['trip'] == 'ENTER') & (orders['status'] == 'FILLED'))
        is_cancelled_entry_order = ((orders['trip'] == 'ENTER') & (orders['status'] == 'CANCELLED'))
        is_filled_exit_lmt_order = (
                    (orders['type'] == 'LMT') & (orders['trip'] == 'EXIT') & (orders['status'] == 'FILLED'))
        is_cancelled_exit_lmt_order = (
                    (orders['trip'] == 'EXIT') & (orders['type'] == 'LMT') & (orders['status'] == 'CANCELLED'))
        is_filled_exit_order = ((orders['trip'] == 'EXIT') & (orders['status'] == 'FILLED'))

        # Check if both of the entry order and the exit limit order are filled. If so, set success to 1, calculate the length of the trade and the rtn
        if any(is_filled_entry_order) and any(is_filled_exit_lmt_order):
            n = calculate_n(is_filled_entry_order, is_filled_exit_lmt_order)
            rtn = calculate_rtn(orders[is_filled_entry_order]['price'].values[0],
                                orders[is_filled_exit_lmt_order]['price'].values[0], n)
            success = 1

        # Check if the entry order is filled and the exit limit order is cancelled. If so, set success to -1, calculate the length of the trade and the rtn. The rtn should use the price of market order.
        elif any(is_filled_entry_order) and any(is_cancelled_exit_lmt_order):
            n = calculate_n(is_filled_entry_order, is_cancelled_exit_lmt_order)
            rtn = calculate_rtn(orders[is_filled_entry_order]['price'].values[0],
                                orders[is_filled_exit_order]['price'].values[0], n)
            success = -1

        # Check if the entry order is cancelled. If so, set success to 0
        elif any(is_cancelled_entry_order):
            n = calculate_n(is_submitted_entry_order, is_cancelled_entry_order)
            rtn = None
            success = 0

        # Set success to null in any other cases
        else:
            n = None
            rtn = None
            success = None

        # Get the asset dt_enter, dt_exit, and the length of the trade.
        asset = orders[orders['trip'] == 'ENTER']['asset'].values[0]
        date_enter = orders[orders['trip'] == 'ENTER']['date'].values[0]
        if any(is_filled_exit_order):
            date_exit = orders[is_filled_exit_order]['date'].values[0]
        else:
            date_exit = None

        # Concat the new record to the ledger
        ledger = pd.concat([ledger, pd.DataFrame({'trade_id': [trade_id],
                                                  'asset': [asset],
                                                  'dt_enter': [date_enter],
                                                  'dt_exit': [date_exit],
                                                  'success': [success],
                                                  'n': [n],
                                                  'rtn': [rtn]})],
                           ignore_index=True)

    return ledger

render_blotter().to_csv('blotter.csv')
blotter_to_ledger(render_blotter()).to_csv('ledger.csv')
print(render_blotter())
print(blotter_to_ledger(render_blotter()))