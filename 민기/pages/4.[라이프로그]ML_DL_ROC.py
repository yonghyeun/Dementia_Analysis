import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

test_roc = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/TEST_ROC.csv')
valid_roc = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/VALID_ROC.csv')

import re

# test_fpr_values = [[float(num) for num in re.findall(r'\d+\.\d+', sublist)] for sublist in test_roc['fpr']]
# test_fpr_values

# test_roc['fpr'] = test_fpr_values


# test_tpr_values = [[float(num) for num in re.findall(r'\d+\.\d+', sublist)] for sublist in test_roc['tpr']]
# test_tpr_values

# test_roc['tpr'] = test_tpr_values


# valid_fpr_values = [[float(num) for num in re.findall(r'\d+\.\d+', sublist)] for sublist in valid_roc['fpr']]
# valid_fpr_values

# valid_roc['fpr'] = valid_fpr_values


# valid_tpr_values = [[float(num) for num in re.findall(r'\d+\.\d+', sublist)] for sublist in valid_roc['tpr']]
# valid_tpr_values

# valid_roc['tpr'] = valid_tpr_values

def mk(a):
    a_1 = re.sub(r'\s+', ' ', a)
    a_2 = a_1.replace(' ', ',')
    a_3 = re.sub(' |\n|\[|\]', '', a_2).split(',')[:-1]
    a_4 = [float(i) for i in a_3]
    return a_4


test_roc['fpr'] = test_roc['fpr'].map(mk)
test_roc['tpr'] = test_roc['tpr'].map(mk)


valid_roc['fpr'] = valid_roc['fpr'].map(mk)
valid_roc['tpr'] = valid_roc['tpr'].map(mk)


# st.dataframe(test_roc)
# st.dataframe(valid_roc)

# import ast


# # 대시보드에 모델별로 ROC 커브 그리기
# for _, row in test_roc.iterrows():
#     model = row['model']
#     fpr = row['fpr']
#     tpr = row['tpr']
#     roc_auc = row['auc']

#     # ROC 커브 그리기
#     fig = px.line(x=fpr, y=tpr, title=f'ROC Curve - {model} (AUC={roc_auc:.2f})')
#     fig.update_layout(
#         xaxis_title='False Positive Rate',
#         yaxis_title='True Positive Rate',
#         showlegend=False
#     )

#     # 대시보드에 그래프 표시
#     st.plotly_chart(fig)

# # 대시보드에 모든 모델의 ROC 커브 그리기
# fig = px.line(title='ROC Curve - All Models')
# fig.update_layout(
#     xaxis_title='False Positive Rate',
#     yaxis_title='True Positive Rate'
# )

# for _, row in test_roc.iterrows():
#     model = row['model']
#     fpr = row['fpr']
#     tpr = row['tpr']
#     roc_auc = row['auc']

#     # ROC 커브 추가
#     fig.add_scatter(x=fpr, y=tpr, name=f'{model} (AUC={roc_auc:.2f})')

# # 대시보드에 그래프 표시
# st.plotly_chart(fig)

# 대시보드에 모든 모델의 ROC 커브 그리기
fig = px.line(title='ROC Curve - All Models(Test)')
fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate'
)

# 진단선 추가
fig.add_shape(
    type='line',
    x0=0,
    y0=0,
    x1=1,
    y1=1,
    line=dict(color='gray', dash='dash'),
    name='Diagonal Line'
)

for _, row in test_roc.iterrows():
    model = row['model']
    fpr = row['fpr']
    tpr = row['tpr']
    roc_auc = row['auc']

    # ROC 커브 추가
    fig.add_scatter(x=fpr, y=tpr, name=f'{model} (AUC={roc_auc:.2f})')

# 대시보드에 그래프 표시
st.plotly_chart(fig)


# 대시보드에 모든 모델의 ROC 커브 그리기
fig = px.line(title='ROC Curve - All Models(Validation)')
fig.update_layout(
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate'
)

# 진단선 추가
fig.add_shape(
    type='line',
    x0=0,
    y0=0,
    x1=1,
    y1=1,
    line=dict(color='gray', dash='dash'),
    name='Diagonal Line'
)

for _, row in valid_roc.iterrows():
    model = row['model']
    fpr = row['fpr']
    tpr = row['tpr']
    roc_auc = row['auc']

    # ROC 커브 추가
    fig.add_scatter(x=fpr, y=tpr, name=f'{model} (AUC={roc_auc:.2f})')

# 대시보드에 그래프 표시
st.plotly_chart(fig)