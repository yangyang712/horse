#払い戻し情報をゲットする

from urllib.request import urlopen
import pandas as pd
import time

def scrape(race_id_list):
    """
    払い戻し表データをスクレイピングする関数

    Parameters:
    ----------
    race_id_list : list
        レースIDのリスト

    Returns:
    ----------
    return_tables_df : pandas.DataFrame
        全払い戻し表データをまとめてDataFrame型にしたもの
    """

    return_tables = {}
    for race_id in race_id_list:
        time.sleep(1)
        try:
            url = "https://db.netkeiba.com/race/" + race_id

            #普通にスクレイピングすると複勝やワイドなどが区切られないで繋がってしまう。
            #そのため、改行コードを文字列brに変換して後でsplitする
            f = urlopen(url)
            html = f.read()
            html = html.replace(b'<br />', b'br')
            dfs = pd.read_html(html)

            print(dfs[1])

            #dfsの1番目に単勝〜馬連、2番目にワイド〜三連単がある
            df = pd.concat([dfs[1], dfs[2]])

            df.index = [race_id] * len(df)
            return_tables[race_id] = df
        except IndexError:
            continue
        except AttributeError: #存在しないrace_idでAttributeErrorになるページもあるので追加
            continue
        except Exception as e:
            print(e)
            break
        except:
            break

    #pd.DataFrame型にして一つのデータにまとめる
    return_tables_df = pd.concat([return_tables[key] for key in return_tables])
    return return_tables_df

#main
race_id_list = []
for place in range(1, 11, 1):#11
    for kai in range(1, 7, 1):#7
        for day in range(1, 13, 1):#13
            for r in range(1, 13, 1):#13
                race_id = "2019" + str(place).zfill(2) + str(kai).zfill(2) + str(day).zfill(2) + str(r).zfill(2)
                race_id_list.append(race_id)

df = scrape(race_id_list)
df.to_csv("harai.csv")