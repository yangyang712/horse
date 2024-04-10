#harai.csvをきれいにする
import pandas as pd
import numpy as np

df = pd.read_csv('harai.csv')
df_new = pd.DataFrame()
df_new['ID'] = df.iloc[:,0]
df_new['ID'] = df_new['ID'].apply(lambda x: str(x))
df_new['kind'] = df.iloc[:,1]
df_new['number'] = df.iloc[:,2]
df_new['harai'] = df.iloc[:,3]
df_new['ninki'] = df.iloc[:,4]

df_new.to_csv('harai1.csv',index=False)