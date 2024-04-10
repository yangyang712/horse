import numpy as np
import pandas as pd
import itertools

#調べたい統計量　人気度ごとの勝率(それぞれの順位について) 人気度と結果の相関

class Simulator:

    kind_names = ['単勝','複勝','枠連','馬連','ワイド','馬単','3連複','3連単']

    def __init__(self,path):
        self.cost = 0
        self.gain = 0
        self.correct_count = 0
        self.total_count = 0
        self.datas = pd.read_csv(path )
        self.datas['ID'] = self.datas['ID'].apply(lambda x: str(x))
        self.datas[['corner1_diff','corner2_diff','corner3_diff','corner4_diff']] = self.datas[['corner1_diff','corner2_diff','corner3_diff','corner4_diff']].apply(lambda x: round(x,2))
        self.id_list = list(set(self.datas['ID'].to_list()))
        #print(self.id_list[:10])
        #print(self.id_list)
        self.races_result = pd.read_csv('harai1.csv' )
        self.races_result['ID'] = self.races_result['ID'].apply(lambda x: str(x))
        #出力用データフレーム
        self.output_data = pd.DataFrame()
        self.output_result = pd.DataFrame()
        #print(self.datas.head())

    def get_race_data(self,id):
        #print(self.datas[self.datas['ID'] == int(id)])
        self.id = id
        self.race_data = self.datas[self.datas['ID'] == id]
        self.output_data = pd.concat([self.output_data,self.race_data])
        return self.race_data
    def get_race_result(self,id):
        self.race_result = self.races_result[self.races_result['ID'] == id]
        self.output_result = pd.concat([self.output_result,self.race_result])
        #単勝の整形
        try:
            self.race_result[self.race_result['kind'] == '単勝'].iat[0,3] = int(self.race_result[self.race_result['kind'] == '単勝'].iat[0,3].replace(',',''))
        except:
            self.race_result[self.race_result['kind'] == '単勝'].iat[0,3] = 100
        #複勝の整形
        df_temp = self.race_result[self.race_result['kind'] == '複勝']
        hukusyo_list = df_temp.iat[0,2].split('br')
        harai_list = df_temp.iat[0,3].split('br')
        ninki_list = df_temp.iat[0,4].split('br')

        for i,number in enumerate(hukusyo_list):
            key = '複勝'
            df = self.race_result[self.race_result['kind'] == '単勝']
            
            df.iat[0,1] = key
            df.iat[0,2] = number
            df.iat[0,3] = int(harai_list[i].replace(',',''))
            df.iat[0,4] = ninki_list[i]
            self.race_result = pd.concat([self.race_result,df])
        #ワイドの整形
        df_temp = self.race_result[self.race_result['kind'] == 'ワイド']
        wide_list = df_temp.iat[0,2].split('br')
        harai_list = df_temp.iat[0,3].split('br')
        ninki_list = df_temp.iat[0,4].split('br')

        for i,number in enumerate(wide_list):
            key = 'ワイド'
            df = self.race_result[self.race_result['kind'] == '単勝']
            
            df.iat[0,1] = key
            df.iat[0,2] = number
            df.iat[0,3] = int(harai_list[i].replace(',',''))
            df.iat[0,4] = ninki_list[i]
            self.race_result = pd.concat([self.race_result,df])
        return self.race_result
    
    def buy_ticket(self,kind_i,number):
        
        gain = self.race_result[(self.race_result['kind'] == Simulator.kind_names[kind_i])&(self.race_result['number'] == str(number))]
        if len(gain) > 0:
            gain = gain.iat[0,3]
            self.gain += int(gain)
            self.correct_count += 1
            print('id:{3}で{0}で{1}にかけて{2}払い戻し'.format( Simulator.kind_names[kind_i],number,gain,self.id))
        self.cost += 100
        self.total_count += 1

    #一番人気の単勝に賭ける
    def buy_all_1(self):
        for i,id in enumerate(self.id_list):
            #必要なレース情報をセットする
            self.get_race_data(id)
            self.get_race_result(id)
            col = self.race_data[self.race_data['人気'] == 1]
            number = col.at[col.index.values[0],'馬番']
            #print(number)
            self.buy_ticket(0,number)
    
    #複勝
    def buy_all_2(self):
        for i,id in enumerate(self.id_list):
            #必要なレース情報をセットする
            self.get_race_data(id)
            self.get_race_result(id)
            #1〜3番人気
            #cols = self.race_data[(self.race_data['人気'] == 1) | (self.race_data['人気'] == 2) | (self.race_data['人気'] == 3) ]
            #1〜2番人気
            cols = self.race_data[(self.race_data['人気'] == 1) | (self.race_data['人気'] == 3) ]
            #print(cols)
            for j in range(len(cols)):

                number = cols.at[cols.index.values[j],'馬番']
                #print(number)
                self.buy_ticket(1,number)

    def buy_all_wide(self):
        for i,id in enumerate(self.id_list):
            #必要なレース情報をセットする
            self.get_race_data(id)
            self.get_race_result(id)
            #1〜3番人気
            cols = self.race_data[ (self.race_data['人気'] == 2) | (self.race_data['人気'] == 3) | (self.race_data['人気'] == 4) ]
            #print(cols)
            numbers = []
            for j in range(len(cols)):
                numbers.append(cols.at[cols.index.values[j],'馬番'])
                #print(number)
            numbers.sort()
            for number in list(itertools.combinations(numbers, 2)):
                self.buy_ticket(4,self.num_to_wide(number))
    def num_to_wide(self,num):
        return str(num[0]) + ' - ' + str(num[1])

    def print_info(self):
        self.output_data.to_csv('temp.csv',index=False)
        self.output_result.to_csv('temp_result.csv',index=False)
        print("総額:{0}円 購入額:{1}円 利益{2}円 回収率{3}% 的中率{4}%".format(self.gain,self.cost,self.gain - self.cost,round(self.gain*100/self.cost,2),round(self.correct_count*100/self.total_count,2)))

if __name__ == '__main__':
    path = 'result2019.csv'
    id = '201902020202'
    simu = Simulator(path)
    #race_data = pd.DataFrame()
    #race_rsult = pd.DataFrame()
    #race_data = simu.get_race_data(id)
    #race_rsult = simu.get_race_result(id)
    #race_data.to_csv('temp.csv',index= False)
    #race_rsult.to_csv('temp_result.csv',index=False)
    #simu.buy_ticket(0,12)
    simu.buy_all_wide()
    simu.print_info()