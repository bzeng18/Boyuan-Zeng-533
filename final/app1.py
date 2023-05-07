from dash import Dash, html, dcc, dash_table, Input, Output, State
import dash_bootstrap_components as dbc
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import os
import refinitiv.dataplatform.eikon as ek
import refinitiv.data as rd
import math
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import itertools

#####################################################

ek.set_app_key(os.getenv('EIKON_APIBZ'))

spacer = html.Div(style={'margin': '10px', 'display': 'inline'})

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

percentage = dash_table.FormatTemplate.percentage(3)

def create_table(id):
    dash_table.DataTable(
        id=id,
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'})
    return dash_table.DataTable

controls = dbc.Card(
    [
        dbc.Row(html.Button('QUERY Refinitiv', id='run-query', n_clicks=0)),

        dbc.Row([
            html.H5('Asset:',
                    style={'display': 'inline-block', 'margin-right': 20}),
            dcc.Input(id='asset', type='text', value="IVV",
                      style={'display': 'inline-block',
                             'border': '1px solid black'}),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α1"), html.Th("n1")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha1',
                                    type='number',
                                    value=-0.01,
                                    max=1,
                                    min=-1,
                                    step=0.001
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n1',
                                    type='number',
                                    value=3,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),

            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("α2"), html.Th("n2")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='alpha2',
                                    type='number',
                                    value=0.01,
                                    max=1,
                                    min=-1,
                                    step=0.001
                                )
                            ),
                            html.Td(
                                dcc.Input(
                                    id='n2',
                                    type='number',
                                    value=5,
                                    min=1,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            ),
            dbc.Table(
                [
                    html.Thead(html.Tr([html.Th("n3")])),
                    html.Tbody([
                        html.Tr([
                            html.Td(
                                dbc.Input(
                                    id='n3',
                                    type='number',
                                    value=30,
                                    max=100,
                                    min=10,
                                    step=1
                                )
                            )
                        ])
                    ])
                ],
                bordered=True
            )
        ]),
        dbc.Row([
            dcc.DatePickerRange(
                id='raw-data-date-picker',
                min_date_allowed = date(2015, 1, 1),
                max_date_allowed = datetime.now(),
                start_date= date(2020, 1, 1)
                #datetime.date(
                #    datetime.now() - timedelta(days=3 * 365)
                #)
            ,
                end_date=date(2023, 4, 13)
                #datetime.now().date()

            )
        ]),
        spacer,
        dbc.Row(html.Button('Submit', id='run-strategy', n_clicks=0)),
        dbc.Row(html.Button('Grid Search', id='search', n_clicks=0))
    ],
    body=True
)

app.layout = dbc.Container(
    [   html.H2('Group Members: Jiarun Wang, Boyuan Zeng: bz100, Yu Yan: yy360',style={'color':'#4169E1'}),
        dbc.Row(
            [
                dbc.Col(controls, md=4),
                #dbc.Col(
                    # Put your reactive graph here as an image!
                 #   md = 8
                #)
            ],
            align="center",
        ),
        html.H2('Historical Data'),
        dash_table.DataTable(
            id="history-tbl",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),
        html.H2('Trade Blotter:'),
        spacer,
        dash_table.DataTable(
            id="orders",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),
        html.H2('Trade Ledger:'),
        spacer,
        dash_table.DataTable(
            id="ledger",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),
        html.H2('Features:'),
        spacer,
        dash_table.DataTable(
            id="features",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),
        html.H2('Prediction:'),
        spacer,
        dash_table.DataTable(
            id="prediction",
            page_action='none',
            style_table={'height': '300px', 'overflowY': 'auto'}
        ),
        dcc.Graph(id="ab-plot"),
        dcc.Store(id='returns'),
        dash_table.DataTable(
        id = "alpha-beta",
        page_action='none',
        style_table={'height': '100px', 'width': '400px','overflowY': 'auto'}
        ),
        dash_table.DataTable(
        id = "alpha-beta-new",
        page_action='none',
        style_table={'height': '100px', 'width': '400px','overflowY': 'auto'}
        ),
        dash_table.DataTable(
        id = "best-params",
        page_action='none',
        style_table={'height': '400px', 'width': '400px','overflowY': 'auto'}
        ),

    ],
    fluid=True
)


@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('asset', 'value'), State('raw-data-date-picker', 'start_date'),
     State('raw-data-date-picker', 'end_date')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, asset, start_date, end_date):
    assets = [start_date, end_date, asset]
    prices, prc_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.OPENPRICE(Adjusted=0)',
            'TR.HIGHPRICE(Adjusted=0)',
            'TR.LOWPRICE(Adjusted=0)',
            'TR.CLOSEPRICE(Adjusted=0)',
            'TR.PriceCloseDate'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )
    prices.rename(
        columns={
            'Open Price': 'open',
            'High Price': 'high',
            'Low Price': 'low',
            'Close Price': 'close'
        },
        inplace=True
    )
    prices.dropna(inplace=True)
    prices.drop(columns='Instrument', inplace=True)

    return (prices.to_dict('records'))

@app.callback(
    Output("orders", "data"),
    Input("run-strategy", "n_clicks"),
    Input("history-tbl", "data"),
    [State('asset','value'),State('alpha1', 'value'), State('n1', 'value'),State('alpha2','value'),State('n2','value')],
    prevent_initial_call=True
)
def render_blotter(n_clicks,history_tbl,asset,alpha1,n1,alpha2,n2):
    prices = pd.DataFrame(history_tbl)
    #1.Get the next business day from Refinitiv!!!!!!!
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date
    rd.open_session()
    next_business_day = rd.dates_and_calendars.add_periods(
        start_date=prices['Date'].iloc[-1].strftime("%Y-%m-%d"),
        period="1D",
        calendars=["USA"],
        date_moving_convention="NextBusinessDay",
    )

    rd.close_session()
    #2. submitted entry orders
    submitted_entry_orders = pd.DataFrame({
        "trade_id": range(1, prices.shape[0]),
        "date": list(pd.to_datetime(prices["Date"].iloc[1:]).dt.date),
        "asset": str(asset),
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(
            prices['close'].iloc[:-1] * (1 + alpha1),
            2
        ),
        'status': 'SUBMITTED'
    })


    # 3. cancelled entry orders
    # if the lowest traded price is still higher than the price you bid, then the
    # order never filled and was cancelled.

    with np.errstate(invalid='ignore'):
        cancelled_entry_orders = submitted_entry_orders[
            np.greater(
                prices['low'].iloc[1:][::-1].rolling(n1).min()[::-1].to_numpy(),
                submitted_entry_orders['price'].to_numpy()
            )
        ].copy()
    cancelled_entry_orders.reset_index(drop=True, inplace=True)
    cancelled_entry_orders['status'] = 'CANCELLED'

    cancelled_entry_orders['date'] = pd.DataFrame(#get the correct cancel date
        {'cancel_date': submitted_entry_orders['date'].iloc[(n1 - 1):].to_numpy()},
        index=submitted_entry_orders['date'].iloc[:(1 - n1 )].to_numpy()
    ).loc[cancelled_entry_orders['date']]['cancel_date'].to_list()

    #4.filled_entry_orders
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
            prices['Date'] == filled_entry_orders['date'].iloc[i]
        )[0]

        slice1 = prices.iloc[idx1:(idx1 + n1)]['low']

        fill_inds = slice1 <= filled_entry_orders['price'].iloc[i]

        if (len(fill_inds) < n1) & (not any(fill_inds)):
            filled_entry_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_entry_orders.at[i, 'date'] = prices['Date'].iloc[
                fill_inds.idxmax()
            ]

    #5 issue exit order upon filled orders
    submitted_exit_orders = filled_entry_orders[filled_entry_orders['status']!='LIVE'].copy()
    submitted_exit_orders['trip'] = 'EXIT'
    submitted_exit_orders['action'] = "SELL"
    submitted_exit_orders['status'] = "SUBMITTED"
    submitted_exit_orders['price'] = submitted_exit_orders['price'] * (1 + alpha2)

    #6 cancel exit orders
    #if the exit order price is greater than the close price of today
    # and greater than the high price of the next n2 days, the order is cancelled
    with np.errstate(invalid='ignore'):
        maxi = list(np.maximum(
            prices['high'][1:][::-1].rolling(n2-1).max()[
            ::-1].to_numpy(), prices[:-1]['close']))
        maxi.append(np.nan)
        iloc = []
        for i in range(len(submitted_exit_orders)):
            row = submitted_exit_orders.iloc[i]
            if row['price']>maxi[list(prices['Date']).index(row['date'])]:
                iloc.append(i)
        cancelled_exit_orders = submitted_exit_orders.iloc[iloc].copy()
    cancelled_exit_orders.reset_index(drop=True, inplace=True)
    cancelled_exit_orders['status'] = 'CANCELLED'
    time_dic = pd.DataFrame(  # get the correct cancel date
        {'cancel_date': prices['Date'].iloc[(n2 - 1):].to_numpy()},
        index=prices['Date'].iloc[:(1 - n2)].to_numpy()
    )
    for i in list(cancelled_exit_orders.index.values):
        cancelled_exit_orders.at[i,'date'] = time_dic.loc[cancelled_exit_orders.at[i,'date']]['cancel_date']

    #7 issue market orders if the order is cancelled
    submitted_market_orders = cancelled_exit_orders.copy()
    submitted_market_orders['trip'] = 'EXIT'
    submitted_market_orders['type'] = 'MARKET'
    submitted_market_orders['action'] = "SELL"
    for i in list(submitted_market_orders.index.values):
        submitted_market_orders.at[i,'price'] = prices[prices['Date']==submitted_market_orders.at[i,'date']]['close']
    submitted_market_orders['status'] = 'SUBMITTED'
    #8 assume the market order is always filled
    filled_market_orders = submitted_market_orders.copy()
    filled_market_orders['status'] = 'FILLED'

    #9 filled exit order
    filled_exit_orders = submitted_exit_orders[
        submitted_exit_orders['trade_id'].isin(
            list(
                set(submitted_exit_orders['trade_id']) - set(
                    cancelled_exit_orders['trade_id']
                )
            )
        )
    ].copy()
    filled_exit_orders.reset_index(drop=True, inplace=True)
    filled_exit_orders['status'] = 'FILLED'
    for i in range(0, len(filled_exit_orders)):

        idx2 = np.flatnonzero(
            prices['Date'] == filled_exit_orders['date'].iloc[i]
        )[0]
        slice2 = prices.iloc[idx2:(idx2 + n2)]['high']
        slice2.iloc[0] = prices.iloc[idx2]['close']
        fill_inds = slice2 >= filled_exit_orders['price'].iloc[i]
        if (len(fill_inds) < n2) & (not any(fill_inds)):
            filled_exit_orders.at[i, 'status'] = 'LIVE'
        else:
            filled_exit_orders.at[i, 'date'] = prices['Date'].iloc[
                fill_inds.idxmax()
            ]

    #10.live entry orders & live exit orders
    live_entry_orders = pd.DataFrame({
        "trade_id": prices.shape[0],
        "date": pd.to_datetime(next_business_day).date(),
        "asset": str(asset),
        "trip": 'ENTER',
        "action": "BUY",
        "type": "LMT",
        "price": round(prices['close'].iloc[-1] * (1 + alpha1), 2),
        'status': 'LIVE'
    },
        index=[0]
    )

    if any(filled_entry_orders['status'] == 'LIVE'):
        live_entry_orders = pd.concat([
            filled_entry_orders[filled_entry_orders['status'] == 'LIVE'],
            live_entry_orders
        ])
        live_entry_orders['date'] = pd.to_datetime(next_business_day).date()

    filled_entry_orders = filled_entry_orders[
        filled_entry_orders['status'] == 'FILLED'
        ]



    if any(filled_exit_orders['status'] == 'LIVE'):
        live_exit_orders = pd.concat([
            filled_exit_orders[filled_exit_orders['status'] == 'LIVE'],
        ])
        live_exit_orders['date'] = pd.to_datetime(next_business_day).date()
        live_entry_orders = pd.concat([live_entry_orders, live_exit_orders])

    filled_exit_orders = filled_exit_orders[
        filled_exit_orders['status'] == 'FILLED'
        ]



    #11. Complete Orders
    orders = pd.concat(
        [
            submitted_entry_orders,
            cancelled_entry_orders,
            filled_entry_orders,
            submitted_exit_orders,
            cancelled_exit_orders,
            submitted_market_orders,
            filled_market_orders,
            filled_exit_orders,
            live_entry_orders,
        ]
    ).sort_values(['trade_id','date'])

    return orders.to_dict('records')

@app.callback(
    Output("ledger", "data"),
    Input("orders", "data"),
    prevent_initial_call=True
)
def blotter_to_ledger(orders):
    blotter = pd.DataFrame(orders)
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

    return ledger.to_dict('records')
@app.callback(
    Output("features", "data"),
    Input("ledger", "data"),
    prevent_initial_call=True
)
def process_feature(ledger):
    df1 = pd.read_excel('hw4_data.xlsx',sheet_name = 'PRICES_DATA')
    df2 = pd.read_excel('hw4_data.xlsx', sheet_name='IVV_DATA')
    prc = pd.DataFrame(ledger)
    # Rename the columns
    df1 = df1.rename(columns={
    'Dates': 'Date',
    'IVV US Equity': 'IVV_US_Equity',
    'IVV AU Equity': 'IVV_AU_Equity',
    'ECRPUS 1Y Index': 'ECRPUS_1Y_Index',
    'SPXSFRCS Index': 'SPXSFRCS_Index',
    'FDTRFTRL Index': 'FDTRFTRL_Index',
    'USCRWTIC Index': 'USCRWTIC_Index',
    'XAU Curncy': 'XAU_Curncy',
    'JPYUSD Curncy': 'JPYUSD_Curncy',
    'DXY Curncy': 'DXY_Curncy',
    'VIX Index': 'VIX_Index'
    })
    df2 = df2.rename(columns={
    'Dates': 'Date'
    })
    prc = prc.rename(columns={
    'dt_enter': 'Date',
    })
    df1['Date'] = pd.to_datetime(df1['Date'])
    df1.loc[78, 'USCRWTIC_Index'] = 1
    prc['Date'] = pd.to_datetime(prc['Date'])

    merged_df = pd.merge(df1, df2, on='Date', how='outer')
    merged_df.drop('FDTRFTRL_Index', axis=1, inplace=True)
    #merged_df.dropna(subset=['success'], inplace=True)
    #merged_df = pd.merge(merged_df, prc, on='Date', how='outer')
    #merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
    exc_list = ['IVV_AU_Equity','IVV_AU_split','Date','IVOL_IMPLIED_FORWARD',
       'IVOL_DELTA']
    for col in merged_df.columns:
        if col not in exc_list:
            merged_df[f'{col}_log_return'] = np.log(merged_df[col]/merged_df[col].shift(1))

    merged_df['AU_log_return'] = np.log(merged_df['IVV_AU_Equity'] *(1/merged_df['IVV_AU_split'])/ merged_df['IVV_AU_Equity'].shift(1))
    merged_df = pd.merge(merged_df, prc, on='Date', how='outer')
    merged_df.dropna(subset=['success'], inplace=True)
    #merged_df['Date'] = merged_df['Date'].dt.strftime('%Y-%m-%d')
    
    merged_df = merged_df[['Date','success','IVV_US_Equity_log_return', 'ECRPUS_1Y_Index_log_return',
       'SPXSFRCS_Index_log_return', 'USCRWTIC_Index_log_return',
       'XAU_Curncy_log_return', 'JPYUSD_Curncy_log_return',
       'DXY_Curncy_log_return', 'VIX_Index_log_return',
       'AU_log_return','IVOL_IMPLIED_FORWARD',
       'IVOL_DELTA']]
    return merged_df.to_dict('records')

def predict(model,X,y,X_test):
    sc = StandardScaler()
    sc.fit(X.values)
    X_std = sc.transform(X.values)
    X_test_std = sc.transform(np.array(X_test.values).reshape(1, -1))
    model.fit(X_std,y)
    y_pred = model.predict(X_test_std)
    return int(float(y_pred[0]))

def new_ledger(features,n3):
    X = features.drop(['Date', 'success'], axis=1)
    y = features['success']
    prediction = []
    for i in range(len(X)-n3):
        X_piece = X.iloc[i:n3+i]
        X_test = X.iloc[n3+i]
        y_piece = y.iloc[i:n3+i]
        model = Perceptron(eta0=0.1)
        y_pred = predict(model, X_piece, np.asarray(y_piece, dtype="|S6"), X_test)
        prediction.append(y_pred)
    return prediction

@app.callback(
    Output("prediction", "data"),
    Input("history-tbl", "data"),
    Input("features", "data"),
    Input("ledger", "data"),
    [State('n3','value')],
    prevent_initial_call=True
)
def prediction(history, features, ledger,n3):
    his = pd.DataFrame(history)
    ledger = pd.DataFrame(ledger)
    features = pd.DataFrame(features)
    his['Date'] = pd.to_datetime(his['Date']).dt.date
    ledger.dropna(subset=['success'], inplace=True)
    ledger = ledger.iloc[n3:]
    #prediction process
    ledger['new_success'] = new_ledger(features,n3)
    ivv_return = []
    for i in range(len(ledger)):
        if ledger['success'].iloc[i]!=0:
            enter = ledger['dt_enter'].iloc[i]
            exit = ledger['dt_exit'].iloc[i]
            n = ledger['n'].iloc[i]
            price_enter = his[his['Date']==pd.to_datetime(enter).date()]['open'].iloc[0]
            price_exit = his[his['Date']==pd.to_datetime(exit).date()]['close'].iloc[0]
            ivv_return.append(math.log(price_exit / price_enter) / n)
        else:
            ivv_return.append(None)
    ledger['ivv_return'] = ivv_return 
    return ledger.to_dict('records')
@app.callback(
    Output("ab-plot", "figure"),
    Output("alpha-beta", "data"),
    Output("alpha-beta-new", "data"),
    Input("prediction", "data"),
    prevent_initial_call=True
)
def render_ab_plot(prediction):
    # Extract columns from the prediction DataFrame
    prediction = pd.DataFrame(prediction)
    success = prediction['success']
    new_success = prediction['new_success']
    
    # Filter data based on condition success != 0
    #filtered_data = prediction[success != 0]
    filtered_data = prediction
    # Create scatter plot with two lines
    fig = px.scatter(filtered_data, x='rtn', y='ivv_return', trendline='ols')
    model = px.get_trendline_results(fig)
    alpha = model.iloc[0]["px_fit_results"].params[0]
    beta = model.iloc[0]["px_fit_results"].params[1]
    new_success_data = filtered_data[new_success == 1]
    
    # Perform regression on filtered data with new_success == 1
    fig_new = px.scatter(new_success_data, x='rtn', y='ivv_return', trendline='ols')
    model_new = px.get_trendline_results(fig_new)
    alpha_new = model_new.iloc[0]["px_fit_results"].params[0]
    beta_new = model_new.iloc[0]["px_fit_results"].params[1]
    regression_line = px.scatter(new_success_data, x='rtn', y='ivv_return', trendline='ols').data[1]
    fig.add_trace(go.Scatter(x=regression_line.x, y=regression_line.y, mode='lines', name='Smart'))
    
    return fig, pd.DataFrame({'old_alpha':[alpha],'old_beta':[beta]}).to_dict('records'),pd.DataFrame({'new_alpha':[alpha_new],'new_beta':[beta_new]}).to_dict('records')

   

@app.callback(
    Output("best-params", "data"),
    Input("search", "n_clicks"),
    [State('asset','value'), State("history-tbl", "data")],
    prevent_initial_call=True
)
def grid_search(n_clicks, asset,history_tbl):
    alpha1_list= [-0.01,-0.005,-0.001,-0.02]
    alpha2_list= [0.01,0.005,0.001,0.02]
    n1_list= [3]
    n2_list= [5]
    n3_list= [50]
    def grid_search(alpha1,alpha2,n1,n2,n3,history_tbll,assetl):
        orders = render_blotter(n_clicks, history_tbll,assetl,alpha1,n1,alpha2,n2)
        ledger = blotter_to_ledger(orders)
        features = process_feature(ledger)
        new_ledger = prediction(history_tbll, features, ledger,n3)
        predict = pd.DataFrame(new_ledger)
        success = predict['success']
        new_success = predict['new_success']
        filtered_data = predict[success != 0]
        fig = px.scatter(filtered_data, x='rtn', y='ivv_return', trendline='ols')
        model = px.get_trendline_results(fig)
        beta = model.iloc[0]["px_fit_results"].params[1]
        new_success_data = filtered_data[new_success == 1]
        fig_new = px.scatter(new_success_data, x='rtn', y='ivv_return', trendline='ols')
        model_new = px.get_trendline_results(fig_new)
        beta_new = model_new.iloc[0]["px_fit_results"].params[1]
        return beta_new-beta
        
    param_combinations = list(itertools.product(alpha1_list, alpha2_list, n1_list, n2_list, n3_list))
    best_beta = []
    for params in param_combinations:
        alpha1, alpha2, n1, n2, n3 = params
        print(params)
        beta_new = grid_search(alpha1,alpha2,n1,n2,n3,history_tbl,asset)
        best_beta.append(beta_new)
    print(param_combinations[best_beta.index(max(best_beta))])
    best_params = param_combinations[best_beta.index(max(best_beta))]
    return pd.DataFrame({'best_params':best_params}).to_dict('records')
if __name__ == '__main__':
    app.run_server(debug=True)
