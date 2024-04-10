from selenium import webdriver
import time
import requests
from bs4 import BeautifulSoup

from selenium.webdriver.chrome.options import Options

#何日に開催されたかを取得する

start_year = 2023
start_month = 4
end_month = 4
end_year = 2023

#年と月からurlを生成
def gen_url(month,year):
    return 'https://race.netkeiba.com/top/calendar.html?year=' + year + '&month=' + month
#startとendから年と月のタプルのリストを作る
def gen_month_list(start_y,start_m,end_y,end_m):
    months = (end_y -start_y  )*12 + end_m - start_m + 1
    #print(months)
    time_list = []
    month = start_m
    year = start_y
    for i in range(months):
        if month == 1:
            year += 1
        time_list.append((str(year),str(month)))
        month = (month % 12) +1
    return time_list
def gen_url_race(id):
    return 'https://race.netkeiba.com/top/race_list.html?kaisai_date=' + id

#メイン関数
"""
time_list = gen_month_list(start_year,start_month,end_year,end_month)
print(time_list)
date_list = []

for t in time_list:
    year,month = t
    url_date = gen_url(month,year)
    print(url_date)
    r = requests.get(url_date)         #requestsを使って、webから取得
    #soup = BeautifulSoup(r.text, "html.parser")
    soup = BeautifulSoup(r.text, 'lxml') #要素を抽出
    soup.find_all('a')


    for a in soup.find_all('a'):
        url = a.get('href')
        if url[:22] == '../top/race_list.html?':
            print('append' + url[-8:])
            date_list.append(url[-8:])
print(date_list)
"""
url = gen_url_race('20240302')
print(url)
options = Options()
#options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable--dev-shm-usage")
driver = webdriver.Chrome(executable_path='./chromedriver',options=options)
driver.get(url)
time.sleep(2)
race_urls = driver.find_elements_by_class_name("RaceList_DataItem")
print(race_urls)