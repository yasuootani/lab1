import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# data_stockを読み込み

data_stock = pd.read_csv("shosenmitsui.csv")
data_stock['datetime_JST'] = pd.to_datetime(data_stock['datetime_JST']).dt.date 

# dataを読み込み
data = pd.read_csv('data4m.csv')
data['Value'] = data['Value'].str.replace(',', '').astype(float)
data["Date"]=pd.to_datetime(data['Date']).dt.date 
# DataFrameを多重配列の形式に変換
multi_array = data  # 最初にタイトル行を追加

multi_array_df = pd.DataFrame(multi_array[0:])
#print(multi_array_df)

# data_stockとmulti_arrayを結合（日付のみで結合）
merged_data = data_stock.merge(multi_array_df, how='left', left_on='datetime_JST', right_on='Date')

# 日付でソート
merged_data = merged_data.sort_values(by='datetime_JST')

# NaNの場合、直近の過去の値を使用
merged_data['Value'] = merged_data['Value'].ffill()

# 不要な列を削除
merged_data=merged_data.iloc[:][["close","datetime_JST","Value"]]
merged_data2=merged_data.dropna()

# 指標データ (特徴量)
X = merged_data2[["Value"]]  
# 株価データ (ターゲット)
y = merged_data2[["close"]]

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

# 線形回帰モデルの初期化
model = LinearRegression()

# モデルの訓練
model.fit(X_train, y_train)

# モデルの評価
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse=mean_squared_error(y_test, y_pred,squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (MSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R^2) Score:", r2)
