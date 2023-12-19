import baostock as bs
import pandas as pd

lg = bs.login()

print('login respond error_code:' + lg.error_code)
print('login respond error_msg:' + lg.error_msg)

# the CSI500 constituent stocks on 2022-04-01
rs = bs.query_zz500_stocks(date='2022-04-01')
zz500_stocks = []
while (rs.error_code == '0') & rs.next():
    zz500_stocks.append(rs.get_row_data())
zz500_df = pd.DataFrame(zz500_stocks, columns=rs.fields)
i=0
dict_sec={}



combined_data_close = pd.DataFrame()
combined_data_open = pd.DataFrame()
combined_data_high = pd.DataFrame()
combined_data_low = pd.DataFrame()
combined_data_volume = pd.DataFrame()
combined_data_amount = pd.DataFrame()


#Below logic is used to create combined datasets of close,open,high low etc, the format will be
"""
time	sh.600006	sh.600008	sh.600021
4/1/2022 10:00	5.34922875	2.96717955	9.56375217
4/1/2022 10:30	5.3787825	2.9394489	9.50391639
4/1/2022 11:00	5.3787825	2.9394489	9.5737248
4/1/2022 11:30	5.35908	2.94869245	9.59367006

above is a sample of output for close prices data

            """

for index, row in zz500_df.iterrows():
    code = row['code']

    rs = bs.query_history_k_data_plus(code,
                                      "date,time,code,open,high,low,close,volume,amount,adjustflag",
                                      start_date='2022-04-01', end_date='2022-07-31',
                                      frequency="30", adjustflag="2 ")
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)

    result['time'] = result['time'].astype(str)

    result['time'] = pd.to_datetime(result['time'].str.slice(0, 14), format='%Y%m%d%H%M%S')

    close_price = result.set_index('time')['close'].rename(code)
    open_price = result.set_index('time')['open'].rename(code)
    high_price = result.set_index('time')['high'].rename(code)
    low_price = result.set_index('time')['low'].rename(code)
    volume = result.set_index('time')['volume'].rename(code)
    amount = result.set_index('time')['amount'].rename(code)

    if combined_data_close.empty:
        combined_data_close = close_price
    else:
        combined_data_close = pd.merge(combined_data_close, close_price, left_index=True, right_index=True, how='outer')


    if combined_data_open.empty:
        combined_data_open = open_price
    else:
        combined_data_open = pd.merge(combined_data_open, open_price, left_index=True, right_index=True, how='outer')


    if combined_data_high.empty:
        combined_data_high = high_price
    else:
        combined_data_high = pd.merge(combined_data_high, high_price, left_index=True, right_index=True, how='outer')


    if combined_data_low.empty:
        combined_data_low = low_price
    else:
        combined_data_low = pd.merge(combined_data_low, low_price, left_index=True, right_index=True, how='outer')


    if combined_data_volume.empty:
        combined_data_volume = volume
    else:
        combined_data_volume = pd.merge(combined_data_volume, volume, left_index=True, right_index=True, how='outer')


    if combined_data_amount.empty:
        combined_data_amount = amount
    else:
        combined_data_amount = pd.merge(combined_data_amount, amount, left_index=True, right_index=True, how='outer')

combined_data_low.to_csv("D:/bao/low.csv")
combined_data_high.to_csv("D:/bao/high.csv")
combined_data_open.to_csv("D:/bao/open.csv")
combined_data_close.to_csv("D:/bao/close.csv")
combined_data_volume.to_csv("D:/bao/volume.csv")
combined_data_amount.to_csv("D:/bao/amount.csv")



# Logout of the system
bs.logout()
