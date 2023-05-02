import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load your data
train = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/train.csv').drop('EMAIL',axis = 1)
test = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/test.csv').drop('EMAIL',axis = 1)

df = pd.concat([train,test],axis = 0)


X = df.drop('target',axis = 1)
Y = df['target']

x_train, x_test, y_train,y_test = train_test_split(X,Y,random_state = 42, test_size = 0.2, stratify = Y)
x_train['target'] = y_train
x_test['target'] = y_test

df = x_train.copy().reset_index(drop = True)
test = x_test.copy().reset_index(drop = True)
# st.write(df)

# 'target' 컬럼의 값을 문자열로 변환
df['target'] = df['target'].astype(str)
# 사용할 컬럼 선택

# 정규화할 컬럼 선택
columns = ['activity_rest', 'activity_score_meet_daily_targets', 'activity_score', 'activity_low',
           'sleep_light', 'sleep_score_latency', 'sleep_midpoint_time', 'sleep_duration',
           'sleep_onset_latency', 'sleep_restless', 'sleep_score_alignment', 'sleep_total',
           'sleep_score_deep', 'sleep_awake']


# 데이터 정규화
scaler = MinMaxScaler()
df[columns] = scaler.fit_transform(df[columns])

# 상태별로 데이터 분리
dem_data = df[df['target'] == 'Dem']
cn_data = df[df['target'] == 'CN']
mci_data = df[df['target'] == 'MCI']

# 레이더 차트 생성
fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=dem_data[columns].mean().values.tolist(),
    theta=columns,
    fill='toself',
    name='Dem'
))

fig.add_trace(go.Scatterpolar(
    r=cn_data[columns].mean().values.tolist(),
    theta=columns,
    fill='toself',
    name='CN'
))

fig.add_trace(go.Scatterpolar(
    r=mci_data[columns].mean().values.tolist(),
    theta=columns,
    fill='toself',
    name='MCI'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 1]
        )
    ),
    showlegend=True
)

# Streamlit에 그래프 표시
st.plotly_chart(fig)