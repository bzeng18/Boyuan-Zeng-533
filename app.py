from dash import *
from dash import Dash, html, dcc, dash_table, Input, Output, State
import refinitiv.dataplatform.eikon as ek
import pandas as pd
import numpy as np
from datetime import datetime, date
import plotly.express as px
import os

ek.set_app_key(os.getenv('EIKON_APIBZ'))

#dt_prc_div_splt = pd.read_csv('unadjusted_price_history.csv')

app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H4('Benchmark',style={'display':'inline-block','margin-right':20}),
        dcc.Input(id = 'benchmark-id', type = 'text', value="IVV"),
        html.H4('Asset',style={'display':'inline-block','margin-right':20}),
        dcc.Input(id = 'asset-id', type = 'text', value="AAPL.O"),
        dcc.DatePickerRange(
            id = "my-date-picker-range",
            min_date_allowed=date(2017, 8, 5),
            max_date_allowed=date(2023, 1, 12),
            start_date=date(2022, 1, 1),
            end_date=date(2023, 1, 1)
        )
    ]),
    html.Button('QUERY Refinitiv', id = 'run-query', n_clicks = 0),
    html.H2('Raw Data from Refinitiv'),
    dash_table.DataTable(
        id = "history-tbl",
        page_action='none',
        style_table={'height': '300px', 'overflowY': 'auto'}
    ),
    html.H2('Historical Returns'),
    dash_table.DataTable(
        id = "returns-tbl",
        page_action='none',
        style_table={'height': '250px', 'overflowY': 'auto'}
    ),
    html.H2('Alpha & Beta Scatter Plot'),
    dcc.DatePickerRange(
        id = "plot-picker-range",
        min_date_allowed=date(2017, 8, 5),
        max_date_allowed=date(2023, 1, 12),
        start_date=date(2022, 1, 1),
        end_date=date(2023, 1, 1)
    ),
    dcc.Graph(id="ab-plot"),
    dcc.Store(id='returns'),
    dash_table.DataTable(
        id = "alpha-beta",
        page_action='none',
        style_table={'height': '100px', 'width': '400px','overflowY': 'auto'}
    ),
    html.P(id='summary-text', children="")
])

@app.callback(
    Output("history-tbl", "data"),
    Input("run-query", "n_clicks"),
    [State('my-date-picker-range', 'start_date'),State('my-date-picker-range', 'end_date'),State('benchmark-id', 'value'), State('asset-id', 'value')],
    prevent_initial_call=True
)
def query_refinitiv(n_clicks, start_date, end_date, benchmark_id, asset_id):
    assets = [benchmark_id, asset_id]
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

    divs, div_err = ek.get_data(
        instruments=assets,
        fields=[
            'TR.DivExDate',
            'TR.DivUnadjustedGross',
            'TR.DivType',
            'TR.DivPaymentType'
        ],
        parameters={
            'SDate': start_date,
            'EDate': end_date,
            'Frq': 'D'
        }
    )

    splits, splits_err = ek.get_data(
        instruments=assets,
        fields=['TR.CAEffectiveDate', 'TR.CAAdjustmentFactor'],
        parameters={
            "CAEventType": "SSP",
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
    prices['Date'] = pd.to_datetime(prices['Date']).dt.date

    divs.rename(
        columns={
            'Dividend Ex Date': 'Date',
            'Gross Dividend Amount': 'div_amt',
            'Dividend Type': 'div_type',
            'Dividend Payment Type': 'pay_type'
        },
        inplace=True
    )
    divs.dropna(inplace=True)
    divs['Date'] = pd.to_datetime(divs['Date']).dt.date
    divs = divs[(divs.Date.notnull()) & (divs.div_amt > 0)]

    splits.rename(
        columns={
            'Capital Change Effective Date': 'Date',
            'Adjustment Factor': 'split_rto'
        },
        inplace=True
    )
    splits.dropna(inplace=True)
    splits['Date'] = pd.to_datetime(splits['Date']).dt.date

    unadjusted_price_history = pd.merge(
        prices, divs[['Instrument', 'Date', 'div_amt']],
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['div_amt'].fillna(0, inplace=True)

    unadjusted_price_history = pd.merge(
        unadjusted_price_history, splits,
        how='outer',
        on=['Date', 'Instrument']
    )
    unadjusted_price_history['split_rto'].fillna(1, inplace=True)

    if unadjusted_price_history.isnull().values.any():
        raise Exception('missing values detected!')

    return(unadjusted_price_history.to_dict('records'))

@app.callback(
    Output("returns-tbl", "data"),
    Input("history-tbl", "data"),
    prevent_initial_call = True
)
def calculate_returns(history_tbl):
    dt_prc_div_splt = pd.DataFrame(history_tbl)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    return(
        pd.DataFrame({
        'Date': numerator[dte_col].reset_index(drop=True),
        'Instrument': numerator[ins_col].reset_index(drop=True),
        'rtn': np.log(
            (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                    denominator[prc_col] * denominator[spt_col]
            ).reset_index(drop=True)
        )
    }).pivot_table(
            values='rtn', index='Date', columns='Instrument'
        ).to_dict('records')
    )

@app.callback(
    Output("ab-plot", "figure"),
    Output("returns", "data"),
    Input("history-tbl", "data"),
    Input("plot-picker-range", "start_date"),
    Input("plot-picker-range", "end_date"),
    Input("my-date-picker-range", "start_date"),
    Input("my-date-picker-range", "end_date"),
    [State('benchmark-id', 'value'), State('asset-id', 'value')],
    prevent_initial_call = True
)
def render_ab_plot(data, start_date, end_date, start_default, end_default,benchmark_id, asset_id):
    if start_date is None:
        start_date = start_default
    if end_date is None:
        end_date = end_default
    if datetime.strptime(start_date, "%Y-%m-%d")<datetime.strptime(start_default, "%Y-%m-%d"):
        start_date = start_default
    if datetime.strptime(end_date, "%Y-%m-%d")>datetime.strptime(end_default, "%Y-%m-%d"):
        end_date = end_default
    dt_prc_div_splt = pd.DataFrame(data)

    # Define what columns contain the Identifier, date, price, div, & split info
    ins_col = 'Instrument'
    dte_col = 'Date'
    prc_col = 'close'
    div_col = 'div_amt'
    spt_col = 'split_rto'

    dt_prc_div_splt[dte_col] = pd.to_datetime(dt_prc_div_splt[dte_col])
    dt_prc_div_splt = dt_prc_div_splt[(dt_prc_div_splt[dte_col]>=datetime.strptime(start_date, "%Y-%m-%d")) & (dt_prc_div_splt[dte_col]<=datetime.strptime(end_date, "%Y-%m-%d"))]
    dt_prc_div_splt = dt_prc_div_splt.sort_values([ins_col, dte_col])[
        [ins_col, dte_col, prc_col, div_col, spt_col]].groupby(ins_col)
    numerator = dt_prc_div_splt[[dte_col, ins_col, prc_col, div_col]].tail(-1)
    denominator = dt_prc_div_splt[[prc_col, spt_col]].head(-1)

    returns = pd.DataFrame({
            'Date': numerator[dte_col].reset_index(drop=True),
            'Instrument': numerator[ins_col].reset_index(drop=True),
            'rtn': np.log(
                (numerator[prc_col] + numerator[div_col]).reset_index(drop=True) / (
                        denominator[prc_col] * denominator[spt_col]
                ).reset_index(drop=True)
            )
        }).pivot_table(
            values='rtn', index='Date', columns='Instrument'
        ).to_dict('records')
    return(px.scatter(returns, x=benchmark_id, y=asset_id, trendline='ols')),returns

@app.callback(
    Output("alpha-beta", "data"),
    Input("returns", "data"),
    [State('benchmark-id', 'value'), State('asset-id', 'value')],
    prevent_initial_call = True
)
def alpha_beta(returns,benchmark_id, asset_id):
    fig = px.scatter(returns, x=benchmark_id, y=asset_id, trendline='ols')
    model = px.get_trendline_results(fig)
    alpha = model.iloc[0]["px_fit_results"].params[0]
    beta = model.iloc[0]["px_fit_results"].params[1]
    return pd.DataFrame({'alpha':[alpha],'beta':[beta]}).to_dict('records')
if __name__ == '__main__':
    app.run_server(debug=True)






