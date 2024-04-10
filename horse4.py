#horseclassの作成
import pandas as pd
import numpy as np

class Horse:
    def __init__(self,name):
        self.name = name
        self.data_all = pd.read_csv('result2019.csv')
        self.horse_data = self.data_all[self.data_all['馬名'] == self.name]
    def info(self):
        filename = 'horse/horse_' + self.name + '.csv'
        filename_d = 'horse/horse_describe' + self.name + '.csv'
        self.horse_data.to_csv(filename,index=False)
        print(self.horse_data.describe())
        self.horse_data.describe().to_csv(filename_d,index=False)

if __name__ == '__main__':
    kiru = Horse('キルロード')
    kiru.info()