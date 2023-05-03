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
train = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/lifelog%20raw%20data/train.csv').drop('EMAIL',axis = 1)
test = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/lifelog%20raw%20data/test.csv').drop('EMAIL',axis = 1)
df = pd.concat([train,test],axis = 0)


X = df.drop('target',axis = 1)
Y = df['target']

x_train, x_test, y_train,y_test = train_test_split(X,Y,random_state = 42, test_size = 0.2, stratify = Y)
x_train['target'] = y_train
x_test['target'] = y_test

df = x_train.copy().reset_index(drop = True)
test = x_test.copy().reset_index(drop = True)




#

target_columns = ['activity_rest', 'activity_score_meet_daily_targets', 'activity_score', 'activity_low',
                  'sleep_light', 'sleep_score_latency', 'sleep_midpoint_time', 'sleep_duration',
                  'sleep_onset_latency', 'sleep_restless', 'sleep_score_alignment', 'sleep_total',
                  'sleep_score_deep', 'sleep_awake']

def create_boxplot(column):
    # Create a DataFrame with the specified column and 'target' columns
    data = pd.DataFrame({column: df[column], 'target': df['target']})

    # Create a box plot using Plotly Express
    fig = px.box(data, x='target', y=column, color='target')

    # Update layout
    fig.update_layout(title=f"Box Plot for {column} by Target")

    # Show the plot in Streamlit
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)

# Assuming you have the DataFrame 'df' containing the data

# Create 7x2 grid layout
columns = st.columns(2)
for i, col in enumerate(target_columns):
    with columns[i % 2]:
        create_boxplot(col)