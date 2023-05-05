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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.io as pio



test_predict = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/test_f1_score.csv').drop(columns='f1_score')
test_label = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/TEST_LABEL.csv')

valid_predict = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/valid_f1_score.csv').drop(columns='f1_score')
valid_label = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/VALID_LABEL.csv')

for i in range(len(test_predict['predict'])):
    test_predict['predict'][i] = list(map(int, test_predict['predict'][i].strip('[]').split(',')))

for i in range(len(valid_predict['predict'])):
    valid_predict['predict'][i] = list(map(int, valid_predict['predict'][i].strip('[]').split(',')))



# actual_labels = ['Real CM', 'Real MCI', 'Real Dem']
# predicted_labels = ['Pred CM', 'Pred MCI', 'Pred Dem']
# names = test_predict['model'].to_list()

# fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(15, 15))
# ax = ax.flatten()

# for i in range(len(test_predict['predict'])):
#     cm = confusion_matrix(test_label['target'], test_predict['predict'][i])
#     sns.heatmap(cm, cbar=False, annot=True, ax=ax[i], linewidths=10, cmap='Blues')
#     ax[i].set_title(f'{names[i]} Confusion Matrix',fontsize = 10)
#     ax[i].set_xticklabels(predicted_labels, rotation=0)
#     ax[i].set_yticklabels(actual_labels, rotation=0)
# fig.suptitle('''TEST DATA에 대한 Confusion Matrix\n''', fontsize=20, weight='bold')
# plt.tight_layout()

# plt.show()




# Define labels and model names
actual_labels = ['Real CM', 'Real MCI', 'Real Dem']
predicted_labels = ['Pred CM', 'Pred MCI', 'Pred Dem']
model_names = test_predict['model'].tolist()

# Create subplot figure
fig = make_subplots(rows=3, cols=3, subplot_titles=model_names)

# Add heatmaps to subplots
for i, model_name in enumerate(model_names):
    cm = confusion_matrix(test_label['target'], test_predict['predict'][i])
    row, col = divmod(i, 3)
    fig.add_trace(go.Heatmap(z=cm, x=predicted_labels, y=actual_labels,
                             colorscale='Blues', showscale=False,
                             hovertemplate='Count: %{z}<extra></extra>'), row=row+1, col=col+1)
    # Add annotations to heatmap cells
    for x in range(len(predicted_labels)):
        for y in range(len(actual_labels)):
            x_center = x # X축 중앙 좌표 계산
            y_center = y # Y축 중앙 좌표 계산
            fig.add_annotation(
                x=x_center,
                y=y_center,
                text=str(cm[y][x]),
                showarrow=False,
                font=dict(color='white' if cm[y][x] > cm.max() / 2 else 'black'),
                xref='x'+str(i+1),
                yref='y'+str(i+1),
                xshift=0,
                yshift=0
            )

fig.update_layout(height=800, width=1000, title_text="Test DATA에 대한 Confusion Matrix",
                  title_font=dict(size=20))

# Convert the figure to JSON
fig_json = fig.to_json()

fig = pio.from_json(fig_json)

# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)


# Define labels and model names
v_actual_labels = ['Real CM', 'Real MCI', 'Real Dem']
v_predicted_labels = ['Pred CM', 'Pred MCI', 'Pred Dem']
model_names = valid_predict['model'].tolist()



# Create subplot figure
fig = make_subplots(rows=3, cols=3, subplot_titles=model_names)

# Add heatmaps to subplots
for i, model_name in enumerate(model_names):
    cm = confusion_matrix(valid_label['target'], valid_predict['predict'][i])
    row, col = divmod(i, 3)
    fig.add_trace(go.Heatmap(z=cm, x=v_predicted_labels, y=v_actual_labels,
                             colorscale='Blues', showscale=False,
                             hovertemplate='Count: %{z}<extra></extra>'), row=row+1, col=col+1)
    # Add annotations to heatmap cells
    for x in range(len(v_predicted_labels)):
        for y in range(len(v_actual_labels)):
            x_center = x # X축 중앙 좌표 계산
            y_center = y # Y축 중앙 좌표 계산
            fig.add_annotation(
                x=x_center,
                y=y_center,
                text=str(cm[y][x]),
                showarrow=False,
                font=dict(color='white' if cm[y][x] > cm.max() / 2 else 'black'),
                xref='x'+str(i+1),
                yref='y'+str(i+1),
                xshift=0,
                yshift=0
            )

fig.update_layout(height=800, width=1000, title_text="Validation DATA에 대한 Confusion Matrix",
                  title_font=dict(size=20))

# Convert the figure to JSON
fig_json = fig.to_json()

fig = pio.from_json(fig_json)

# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)


