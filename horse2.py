import requests
from bs4 import BeautifulSoup
import pandas as pd
import pyparsing as pp
import numpy as np
import math

#cornerdiffについては元の情報もcsvに保存しておく


# DataFrame列名定義
columns = ['diff', 'horse_no']

# 差の定数(unit:馬身)
DIFF_GROUP = 0.3
DIFF_MIN = 1.5
DIFF_MID = 3.0
DIFF_MUCH = 6.0

class ParsePass():
    
    def __init__(self):
        
        # 馬番
        horse_no = pp.Word(pp.nums).setParseAction(self._horse_no_action)
        
        # 馬群
        group = pp.Suppress(pp.Literal('(')) + \
                    pp.Optional(pp.delimitedList(pp.Word(pp.nums), delim=',')) + \
                    pp.Suppress(pp.Literal(')'))
        group.ignore('*')
        group.setParseAction(self._group_action)

        # 情報要素
        element = (group | horse_no)
        
        # 前走馬との差
        diff_min = pp.Suppress(pp.Optional(pp.Literal(','))).setParseAction(self._diff_min_action) + element
        diff_mid = pp.Suppress(pp.Literal('-')).setParseAction(self._diff_mid_action) + element
        diff_much = pp.Suppress(pp.Literal('=')).setParseAction(self._diff_much_action) + element

        # 全体定義
        self._passing_order = element + pp.ZeroOrMore( diff_mid | diff_much | diff_min )
        
    def _horse_no_action(self, token):
        
        self._data = self._data.append({'diff':self._diff, 'horse_no':token[0]}, ignore_index=True)
        return

    def _group_action(self, token):
        
        for no in token:
            self._data = self._data.append({'diff':self._diff, 'horse_no':no}, ignore_index=True)
            self._diff += DIFF_GROUP
        self._diff -= DIFF_GROUP
        return
        
    def _diff_min_action(self, token):
        
        self._diff += DIFF_MIN
        return
        
    def _diff_mid_action(self, token):
        
        self._diff += DIFF_MID
        return
    
    def _diff_much_action(self, token):
        
        self._diff += DIFF_MUCH
        return
        
    def parse(self, pass_str):
        
        # 初期化
        self._data = pd.DataFrame(columns=columns)
        self._diff = 0
        # parse

        self._passing_order.parseString(pass_str)
        # index調整
        #self._data.index = np.arange(1, len(self._data)+1)
        #self._data.index.name = 'rank'
        
        return self._data



def get_race_dfs(id):
    url = "https://race.netkeiba.com/race/result.html?race_id=" +id + "&rf=race_list"
    dfs = pd.read_html(url)
    return dfs


#https://race.netkeiba.com/race/result.html?race_id=201901010612&rf=race_list
def get_race_tyo_dfs(id):
    url = "https://race.netkeiba.com/race/oikiri.html?race_id=" + id + "&rf=race_submenu"
    dfs = pd.read_html(url)
    #print(len(dfs))
    return dfs

def get_race_condition(df,id):
    url = "https://race.netkeiba.com/race/result.html?race_id=" +id + "&rf=race_list"
    r = requests.get(url)         #requestsを使って、webから取得
    #soup = BeautifulSoup(r.content, "html.parser")
    soup = BeautifulSoup(r.content, 'lxml') #要素を抽出
    racedata = soup.find('div', class_='RaceData01')
    racename = soup.find('div', class_='RaceName')
    #print(racedata.text.split())
    racedata_list = racedata.text.split()
    #print(racename.contents[0].strip())
    df['racename'] = racename.contents[0].strip()
    df['racestart'] = racedata_list[0]
    df['racekind'] = racedata_list[2] + racedata_list[3]
    df['raceweather'] = racedata_list[-3]
    df['racefieldconditin'] = racedata_list[-1]
    #print(df)
    return df


def get_race_result(dfs,id):
    race_result = dfs[0].copy()
    race_result["ID"] = id
    return race_result

def get_corner_rank(dfs):
    race_corner_rank = dfs[3].copy()
    return race_corner_rank
#class化したほうが良い
def get_race_csv(id):

    try:


        dfs = get_race_dfs(id)
        df = get_race_result(dfs,id)
        #print(df)

        df_c_rank = get_corner_rank(dfs)
        #df_c_rank.to_csv("c_rank.csv")
        pass_data = df_c_rank.iloc[:,1]

        #r = requests.get(url)         #requestsを使って、webから取得
        #soup = BeautifulSoup(r.text, "html.parser")
        #soup = BeautifulSoup(r.text, 'lxml') #要素を抽出
        #print(soup)
        #着順とウマ番号の辞書
        rank_horse_dict = df["馬番"].to_dict()
        rank_horse_dict = {str(v): k for k, v in rank_horse_dict.items()}

        corner_names = ['corner1_diff','corner2_diff','corner3_diff','corner4_diff']
        pass_parsing = ParsePass()
        for i,pass_str in enumerate(pass_data):
            #print(id,pass_str)
            if pd.isna(pass_str):
                co_data = pd.DataFrame(data = np.zeros(df.shape[0]))
            else:
                co_data = pass_parsing.parse(pass_str)
                #print(type(co_data['horse_no'].iloc[2]))
                #co_data.to_csv("corner.csv")
                co_data = co_data.sort_values('horse_no', key=lambda col: col.map(rank_horse_dict),ignore_index=True)
            df[corner_names[i]] = co_data.iloc[:,0]


        dfs_tyo = get_race_tyo_dfs(id)[0]
        rank_horse_dict = {int(v): k for v,k in rank_horse_dict.items()}
        dfs_tyo = dfs_tyo.sort_values('馬番', key=lambda col: col.map(rank_horse_dict),ignore_index=True)
        df['condition'] = dfs_tyo['評価.1']
        df = get_race_condition(df,id)
        #df.to_csv("result.csv")
        return df
    except:
        return pd.DataFrame()

#id = '201901010102'
df = pd.DataFrame()

#df = get_race_csv(id)
#df = get_race_condition(df,id)

#df = pd.concat([df,get_race_csv(id)])
#df.to_csv("result.csv")
#201910021212までは調べてる
race_id_list = []
for place in range(1, 11, 1):#11
    for kai in range(1, 7, 1):#7
        for day in range(1, 13, 1):
            for r in range(1, 13, 1):
                race_id = "2020" + str(place).zfill(2) + str(kai).zfill(2) + str(day).zfill(2) + str(r).zfill(2)
                race_id_list.append(race_id)
for id in race_id_list:
    df = pd.concat([df,get_race_csv(id)])
    df.to_csv("result.csv")

df.to_csv("result.csv")