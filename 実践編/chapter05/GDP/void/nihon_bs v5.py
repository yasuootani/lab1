import csv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # バックエンドを設定
import matplotlib.pyplot as plt
from matplotlib import rc
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# LaTeXフォントを使用
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#plt.rcParams['text.ipaxgothc'] = True

def read_csv_to_numpy(filename):
    with open(filename, encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader)  # ヘッダー行を取得
        data = np.array([row for row in reader])  # データ行をNumPy配列に変換
    return header, data

header, data = read_csv_to_numpy('GDP/data/nihon_bs.csv')

nendo = [header[1], header[2]]
data_2003 = np.array([data[0][1], data[1][1], data[2][1], data[3][1], data[9][1], data[10][1]], dtype=np.int32)
data_2021 = np.array([data[0][2], data[1][2], data[2][2], data[3][2], data[9][2], data[10][2]], dtype=np.int32)
keta = np.array([100000] * 6, dtype=np.int32)
data_2003 = data_2003 / keta
data_2021 = data_2021 / keta

# グラフの初期化
fig, ax = plt.subplots()

# マイナスからスタートするための初期値
total1 = total2 = 0
bottom1 = bottom2 = 0

# 年度1の滝グラフを作成
fig = make_subplots(rows=1, cols=1)
trace_2003=go.Waterfall(
    x=[data[0][0], data[1][0], data[2][0], data[3][0], data[9][0], data[10][0]],
    y=data_2003,
    measure=['absolute', 'relative', 'relative', 'relative', 'relative', 'total'],
    name='2003増減',
    decreasing = dict(marker={'color': 'lightgrey'}),
    increasing = dict(marker={'color': 'lightblue'}),
    totals = dict(marker={'color': 'royalblue'}),
    connector = {'line': {'width': 1, 'color':'lightgrey', 'dash': 'solid'}}
    
)

fig.add_trace(trace_2003)

trace_2021=go.Waterfall(
    x=[data[0][0], data[1][0], data[2][0], data[3][0], data[9][0], data[10][0]],
    y=data_2021,
    measure=['absolute', 'relative', 'relative', 'relative', 'relative', 'total'],
    name='2021増減',
    decreasing = dict(marker={'color': 'lightpink'}),
    increasing = dict(marker={'color': 'pink'}),
    totals = dict(marker={'color': 'red'}),
    connector = {'line': {'width': 1, 'color':'lightgrey', 'dash': 'solid'}}
)

fig.add_trace(trace_2021)



# グラフの装飾
fig.update_layout(
    title='日本の「債務超過」',
    xaxis_title='年度',

    yaxis_title='兆円'
)


fig.show()

input("Press Enter to exit...")


#decreasing = dict(marker={'color': 'lightgrey'}),
#    increasing = dict(marker={'color': 'lightblue'}),
#    totals = dict(marker={'color': 'royalblue'}),
#    connector = {'line': {'width': 1, 'color':'lightgrey', 'dash': 'solid'}}