import baostock as bs
import pandas as pd
from googletrans import Translator

#index composition of csi 500 index on 2021-01-01
lg = bs.login()
if lg.error_code == '0':
    print('Login successful')
else:
    print(f'Login failed with error: {lg.error_msg}')

rs = bs.query_zz500_stocks(date='2021-01-01')
print('query_zz500 error_code:'+rs.error_code)
print('query_zz500 error_msg:'+rs.error_msg)

zz500_stocks = []
while (rs.error_code == '0') & rs.next():
    zz500_stocks.append(rs.get_row_data())

result = pd.DataFrame(zz500_stocks, columns=rs.fields)

#
# result.to_csv("D:/bao/zz500_stocks.csv", encoding="gbk", index=False)
print(result)



