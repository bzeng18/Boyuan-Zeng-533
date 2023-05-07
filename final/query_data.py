import eikon as ek
import pandas as pd
from datetime import datetime
import numpy as np
import os

ek.set_app_key(os.getenv('EIKON_APIBZ'))

def query_refinitiv():

    # Define the asset to retrieve data for
    asset = 'IVV'
    # Define the date range as strings
    start_date = '2020-01-01'
    end_date = '2023-04-13'

    # Retrieve the pricing data for the asset and date range
    prices, prc_err = ek.get_data(
        instruments=asset,
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

    # Rename the columns of the DataFrame to more descriptive names
    prices.rename(
        columns={
            'TR.OPENPRICE(Adjusted=0)': 'open',
            'TR.HIGHPRICE(Adjusted=0)': 'high',
            'TR.LOWPRICE(Adjusted=0)': 'low',
            'TR.CLOSEPRICE(Adjusted=0)': 'close',
            'TR.PriceCloseDate': 'date'
        },
        inplace=True
    )

    # Convert the date column to a pandas datetime format
    prices['Date'] = pd.to_datetime(prices['Date'])

    # Format the date column as strings in the format "YYYY-MM-DD"
    prices['Date'] = prices['Date'].dt.strftime('%Y-%m-%d')

    print(prices)

    # Save the prices DataFrame to a CSV file named 'prices.csv' in the current directory
    prices.to_csv('prices.csv', index=False)

    return prices


query_refinitiv()