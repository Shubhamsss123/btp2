import pandas as pd 

pd.read_excel('discharge data.xls').to_csv('data.csv',index=False)