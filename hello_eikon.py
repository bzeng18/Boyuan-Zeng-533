import os
import eikon as ek
eikon_api = os.getenv('EIKON_APIBZ')
ek.set_app_key(eikon_api)

df = ek.get_timeseries(["MSFT.O"],start_date = "2016-01-01", end_date="2016-01-10")

df
