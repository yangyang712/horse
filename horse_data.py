#csvファイルを整形する
import pandas as pd 
import numpy as np
import re
#タイムが欠損しているものについてはdrop
#コーナー通過順位は再度取得し直したい

class Horse_Data:
    def __init__(self,filename):
        self.data = pd.read_csv(filename)
    def clean(self):
        self.data['ID'] = self.data['ID'].apply(lambda x: str(x))
        #性齢を整形
        self.data = self.data.dropna(subset=['タイム'])
        self.data = self.data.fillna(0)
        self.data['gender'] = self.data['性齢'].apply(lambda x:self.seirei(x,0))
        self.data['age'] = self.data['性齢'].apply(lambda x:self.seirei(x,1))
        #ゴールタイムをパース
        self.data['goal_time'] = self.data['タイム'].apply(lambda x: self.parse_time(x))
        #コーナー通過順位をパース
        cornar_names = ['corner_rank1','corner_rank2','corner_rank3','corner_rank4']
        #print(self.data['コーナー通過順'].apply(lambda x:self.parse_corner_rank(x)))
        for i,corner in enumerate(cornar_names):
            self.data[corner] = self.data['コーナー通過順'].apply(lambda x:self.parse_corner_rank(x,i))
        #馬体重(増減)をパース
        self.data['weight'] = self.data['馬体重(増減)'].apply(lambda x:self.parse_weight(x,0))
        self.data['weight_vari'] = self.data['馬体重(増減)'].apply(lambda x:self.parse_weight(x,1))
        #着差をパースする
        #self.id_list = list(set(self.data['ID'].to_list()))
        #self.diff_arrive()
    #牝０牡１セ２
    def seirei(self,string,mode):
        if mode == 0:
            if string[0] == '牝':
                return 0
            elif string[0] == '牡':
                return 1
            else:
                return 2
        else:
            return int(string[1])
    #ゴールタイムのパース
    def parse_time(self,string):
        weight = [60,1,0.1]
        #print(string)
        goal_time = 0
        try:
            time_list = re.findall('[0-9]+',string)
        except:
            time_list = []
        for i,time in enumerate(time_list):
            goal_time += int(time) * weight[i]
        return goal_time
    #コーナー通過順位をパース 少し非効率
    def parse_corner_rank(self,string,index):
        rank_list = re.findall('[0-9]+',string)
        rank_list = self.fill_list(rank_list)
        return rank_list[index]
    #リストを埋める
    def fill_list(self,list_input):
        for i in range(4-len(list_input)):
            list_input.insert(0,0)
        return list_input
    #体重をパース
    def parse_weight(self,string,index):
        if index == 0:
            weight_result = re.findall('[0-9]+',string)
            return int(weight_result[0])
        else:
            weight_result = re.findall('[+|-][0-9]+',string)
            if len(weight_result) > 0:
                weight_result = int(weight_result[0])
            else:
                weight_result = 0
        return weight_result
    #着差をパース
    def diff_arrive(self):
        result = pd.DataFrame()
        for id in self.id_list:
            race_data = self.data[self.data['ID'] == id]
            diff_now = 0
            diff_list = race_data['着差'].to_list()
            for diff in diff_list:
                diff_now += self.parse_diff(diff)
                temp = pd.DataFrame(data=[[str(id),round(diff_now,2)]],columns=['ID','diff_arrive'])
                result = pd.concat([result,temp])
        #self.data = pd.merge(self.data,result,on='ID')
        self.data['diff_arrive'] = result['diff_arrive']
        #result.to_csv('kesu.csv',index=False)
    def parse_diff(self,input):
        diff_dict = {'アタマ':0.2,'クビ':0.3,'ハナ': 0.1,'大':11,'同着':0,'2位降着':0,'1位降着':0,'3位降着':0,'4位降着':0,'5位降着':0}
        if ('/' in str(input)) and (input not in diff_dict) :
            dig1 = re.findall('[0-9]/[0-9]',input)[0]
            dig2 = re.findall('[0-9][.]',input)
            #print(dig1)
            result = int(dig1[0]) / int(dig1[-1])
            if len(dig2) > 0:
                result += int(dig2[0][0])
        elif input in diff_dict.keys():
            result = diff_dict[input]
        else:
            result = int(input)
        return result
            


    #結果を出力する
    
    def output(self):
        self.data.to_csv('result2019_seikei.csv',index=False)

if __name__ == '__main__':
    cleaner = Horse_Data('result2019.csv')
    cleaner.clean()
    cleaner.output()