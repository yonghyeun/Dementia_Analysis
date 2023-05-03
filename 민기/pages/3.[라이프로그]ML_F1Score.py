import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots


df_test = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/test_f1_score.csv')
df_valid = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/valid_f1_score.csv')
df_test = df_test.sort_values(by = 'f1_score', ascending = False)
df_valid = df_valid.sort_values(by = 'f1_score', ascending = False)
# Data


def main():

    # Create the bar graph
    fig = px.bar(df_test, x='f1_score', y='model', orientation='h', color='model',
             labels={'f1_score': 'F1 Score', 'model': 'Model'})

# Update layout
    fig.update_layout(title='Test set F1 Scores for Different Models',
                  xaxis=dict(title='f1_score'),
                  yaxis=dict(title='Model'))

    st.plotly_chart(fig)

if __name__ == '__main__':
    main()


def main():

    # Create the bar graph
    fig = px.bar(df_valid, x='f1_score', y='model', orientation='h', color='model',
             labels={'f1_score': 'F1 Score', 'model': 'Model'})

# Update layout
    fig.update_layout(title='Validation set F1 Scores for Different Models',
                  xaxis=dict(title='f1_score'),
                  yaxis=dict(title='Model'))

    st.plotly_chart(fig)

if __name__ == '__main__':
    main()