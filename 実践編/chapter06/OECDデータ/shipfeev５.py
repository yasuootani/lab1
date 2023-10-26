from selenium import webdriver
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.common.by import By
options = webdriver.EdgeOptions()
service = EdgeService(executable_path='C:\\Users\\ip2305\\Desktop\\msedgedriver.exe')
driver = webdriver.Edge(service=service, options=options)
import numpy as np
import pandas as pd
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

driver.get("https://stock-marketdata.com/china-containerized-freight-index")

# JavaScriptを使ってクリックイベントをトリガーdocument.querySelector("#post-43052 > div > div:nth-child(75) > label")
script = """
var element = document.querySelector("#post-43052 > div > div:nth-child(75) > label");
element.click();
"""
driver.execute_script(script)

script = """
var element = document.querySelector('#post-43052 > div > div:nth-child(77) > label');
element.click();
"""
driver.execute_script(script)

script = """
var element = document.querySelector("#post-43052 > div > div:nth-child(79) > label");
element.click();
"""
driver.execute_script(script)

#element = driver.find_element(By.XPATH, '//html/body/div[1]/div[4]/div/main/article/div/div[8]/label')

# クリック→クリックできずデータが取れない
#action = ActionChains(driver)
#action.move_to_element(element).click().perform()

#element = driver.find_element(By.XPATH, '/html/body/div[1]/div[4]/div/main/article/div/div[7]/label')
#クリック
#action = ActionChains(driver)
#action.move_to_element(element).click().perform()

#element = driver.find_element(By.XPATH, '/html/body/div[1]/div[4]/div/main/article/div/div[8]/label')
#クリック
#action = ActionChains(driver)
#action.move_to_element(element).click().perform()

table_xpaths = ['//html//body//div[1]//div[4]//div//main//article//div//figure[2]//div//table','//html//body//div[1]//div[4]//div//main//article//div//div[6]//div//figure//div//table',
'//html//body//div[1]//div[4]//div//main//article//div//div[7]//div//figure//div//table',
'//html//body//div[1]//div[4]//div//main//article//div//div[8]//div//figure//div//table']

data = []  # テーブルデータを格納するためのリスト

# ページ内のテーブルを取得
for table_xpath in table_xpaths:
    table = driver.find_element(By.XPATH, table_xpath)
    
    # テーブルの各行を取得し、行ごとのデータをリストに追加していく
    rows = table.find_elements(By.TAG_NAME, 'tr')
    table_data = []  # 各テーブルのデータを格納するためのリスト
    for row in rows:
        columns = row.find_elements(By.TAG_NAME, 'td')
        row_data = []  # 各行のデータを格納するためのリスト
        for column in columns:
            # テキストデータを取得してリストに追加
            row_data.append(column.text)
        table_data.append(row_data)  # 行データをテーブルデータリストに追加
    data.append(table_data)  # テーブルデータをデータリストに追加

# テーブルデータをNumPyの配列に変換
df = pd.DataFrame(data)

# NumPyの配列を表示
print(df)

driver.quit()  # ブラウザを閉じる



#'\\html\\body\\div[1]\\div[4]\\div\\main\\article\\div\\div[6]\\div\\figure\\div\\table',
#'\\html\\body\\div[1]\\div[4]\\div\\main\\article\\div\\div[7]\\div\\figure\\div\\table',
#'\\html\\body\\div[1]\\div[4]\\div\\main\\article\\div\\div[8]\\div\\figure\\div\\table'

#'\\html\\body\\div[1]\\div[4]\\div\\main\\article\\div\\div[6]\\label'
#'\\html\\body\\div[1]\\div[4]\\div\\main\\article\\div\\div[7]\\label'
#'\\html\\body\\div[1]\\div[4]\\div\\main\\article\\div\\div[8]\\label'
#//*[@id="post-43052"]/div/div[6]/div/figure/div/table/tbody


#/html/body/div[1]/div[4]/div/main/article/div/div[6]/div
#/html/body/div[1]/div[4]/div/main/article/div/div[7]/div
