import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn,optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchinfo import summary
from torch.autograd import Variable
import tensorflow as tf
#from tensorflow import keras 
#from keras import metrics

hyoka=tf.keras.metrics.RootMeanSquaredError(name='rmse')

# data_stockを読み込み

data_stock = pd.read_csv("shosenmitsui.csv")
data_stock['datetime_JST'] = pd.to_datetime(data_stock['datetime_JST']).dt.date 

# dataを読み込み
data = pd.read_csv('data.csv')
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
merged_data['Value'] = merged_data['Value'].fillna(method='ffill')

# 不要な列を削除
merged_data=merged_data.iloc[:][["close","datetime_JST"]]
merged_data2=merged_data.dropna()
# NaNの場合、直近の過去の値を使用
#for index, row in merged_data.iterrows():
##    if row['値']=="":
#        merged_data.at[index, '値'] = initial_value
##   else:
#        initial_value = row['値']

print (merged_data2)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(merged_data2[['close']])

# テスト用と訓練用で分割
df_train, df_test = train_test_split(df_scaled, test_size=0.3, shuffle=False)

window_size = 5
n_data = len(merged_data2) - window_size + 1 -1

n_dim = df_train.shape[1]
n_train = len(df_train) - window_size + 1 - 1
n_test = len(df_test) - window_size + 1 - 1

# 正解データを準備
train = np.zeros((n_train, window_size, n_dim))
train_labels = np.zeros((n_train, n_dim))
for i in range(n_train):
    train[i] = df_train[i:i+window_size]
    train_labels[i] = df_train[i+window_size]

# テストデータを準備
test = np.zeros((n_test, window_size, n_dim))
test_labels = np.zeros((n_test, n_dim))
for i in range(n_test):
    test[i] = df_test[i:i+window_size]
    test_labels[i] = df_test[i+window_size]

# 訓練ラベルの用意。今回は株価を予測する
train_labels =train_labels[:, 0]

#pytorchのtensor形式に変換
train = torch.tensor(train, dtype=torch.float)
labels = torch.tensor(train_labels, dtype=torch.float)
dataset = torch.utils.data.TensorDataset(train, labels)
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 多変量を入力して、１変数の予測結果を返すLSTNモデル.
class MyLSTM(nn.Module):
    def __init__(self, feature_size, hidden_dim, n_layers):
        super(MyLSTM, self).__init__()

        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_output = 1

        self.lstm = nn.LSTM(feature_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.n_output)

    def forward(self, x):
        # hidden state
        h_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim))
        # cell state
        c_0 = Variable(torch.zeros(self.n_layers, x.size(0), self.hidden_dim))
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) # (input, hidden, and internal state)
        hn = hn.view(-1, self.hidden_dim) 
        y = self.fc(hn)
        y = y.reshape(self.n_output, -1)

        return y


feature_size  = 1
n_hidden  = 64
n_layers  = 1

net = MyLSTM(feature_size, n_hidden, n_layers)

summary(net)
#parameter 設定
func_loss = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_history = []
device = torch.device("cuda:0" if torch.cuda. is_available() else "cpu")
epochs = 50

net.to(device)

#学習を実施

for i in range(epochs+1):
    net.train()
    tmp_loss = 0.0
    for j, (x, t) in enumerate(train_loader):
        x = x.to(device)
        optimizer.zero_grad()
        y = net(x) 
        y = y.to('cpu')
        loss = func_loss(y, t)
        loss.backward()
        optimizer.step() 
        tmp_loss += loss.item()
    tmp_loss /= j+1
    loss_history.append(tmp_loss)
    print('Epoch:', i, 'Loss_Train:', tmp_loss)

# 損失関数を描く

plt.plot(range(len(loss_history)), loss_history, label='train')
plt.legend()

plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

predicted_train_plot = []
predicted_test_plot =[]
net.eval()

for k in range(n_test):
    x = torch.tensor(test[k])
    x = x.reshape(1, window_size, feature_size)
    x = x.to(device).float()
    y = net(x)
    y = y.to('cpu')
    predicted_test_plot.append(y[0].item())

#testデータの予測結果
#     
plt.plot(range(len(df_test)), df_test[:, 0], label='Correct')
plt.plot(range(window_size, window_size+len(predicted_test_plot)), predicted_test_plot, label='Test result')
plt.legend()
plt.show()

print(hyoka(df_test[5:,0],predicted_test_plot))