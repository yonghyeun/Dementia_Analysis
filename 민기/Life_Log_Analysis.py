import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.subplots as sp
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import re



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

# st.write(df)


# 정규화할 컬럼 선택
columns = ['activity_rest', 'activity_score_meet_daily_targets', 'activity_score', 'activity_low',
           'sleep_light', 'sleep_score_latency', 'sleep_midpoint_time', 'sleep_duration',
           'sleep_onset_latency', 'sleep_restless', 'sleep_score_alignment', 'sleep_total',
           'sleep_score_deep', 'sleep_awake']

# 레이더 차트
def main():
    st.title('Rader Chart')

   

    col1, col2 = st.columns([1,1])  # 첫 번째 컬럼은 두 번째 컬럼보다 2배 너비로 설정합니다.

    with col1:

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
                     showline=False,
                    range=[0, 1],
                    tickfont=dict(size=10)
                )
            ), legend=dict(
                    x=1,  # 범례 위치를 우측으로 조정
                    y=0,  # 범례 위치를 하단으로 조정
                    bgcolor='rgba(255, 255, 255, 0.5)',  # 범례 배경색 설정
                    bordercolor='rgba(0, 0, 0, 0.5)'  # 범례 테두리 색상 설정
            ),
            showlegend=True,
            font=dict(
                size=17  # 변수 이름의 글꼴 크기 설정
            )
        )

        # Streamlit에 그래프 표시
        st.plotly_chart(fig)
    with col2:
        st.write("""
        #### CN(정상)와 MCI(경도인지장애)는 거의 동일 양상을 보이지만 Dem(치매)의 경우에는 전혀 다른 양상을 볼 수있다.
        """)
if __name__ == '__main__':
    main()


# 박스플롯
# target_columns = ['activity_rest', 'activity_score_meet_daily_targets', 'activity_score', 'activity_low',
#                   'sleep_light', 'sleep_score_latency', 'sleep_midpoint_time', 'sleep_duration',
#                   'sleep_onset_latency', 'sleep_restless', 'sleep_score_alignment', 'sleep_total',
#                   'sleep_score_deep', 'sleep_awake']

# def create_boxplot(column):
#     # Create a DataFrame with the specified column and 'target' columns
#     data = pd.DataFrame({column: df[column], 'target': df['target']})

#     # Create a box plot using Plotly Express
#     fig = px.box(data, x='target', y=column, color='target')

#     # Update layout
#     fig.update_layout(title=f"Box Plot for {column} by Target")

#     # Show the plot in Streamlit
#     col1, col2 = st.columns(2)
#     with col1:
#         st.plotly_chart(fig)

# # Assuming you have the DataFrame 'df' containing the data

# # Create 7x2 grid layout
# columns = st.columns(2)
# for i, col in enumerate(target_columns):
#     with columns[i % 2]:
#         create_boxplot(col) 
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
    st.plotly_chart(fig)

def main():
    st.title('Box Plots')

    # Assuming you have the DataFrame 'df' containing the data

    # Create a multiselect widget for selecting multiple columns
    selected_columns = st.multiselect('Select Columns', target_columns)

    # Calculate the number of columns
    num_columns = len(selected_columns)

    # Display the selected charts in a 1-column and 2-column layout
    for i, column in enumerate(selected_columns):
        if i % 2 == 0:
            col1, col2 = st.columns([1, 1])
            with col1:
                create_boxplot(column)
        else:
            with col2:
                create_boxplot(column)

if __name__ == '__main__':
    main()



# # 정규화할 컬럼 선택
# columns = ['activity_rest', 'activity_score_meet_daily_targets', 'activity_score', 'activity_low',
#            'sleep_light', 'sleep_score_latency', 'sleep_midpoint_time', 'sleep_duration',
#            'sleep_onset_latency', 'sleep_restless', 'sleep_score_alignment', 'sleep_total',
#            'sleep_score_deep', 'sleep_awake']

# # 레이더 차트
# def main():


   

#     col1, col2 = st.columns([1,1])  # 첫 번째 컬럼은 두 번째 컬럼보다 2배 너비로 설정합니다.

#     with col1:

#         # 데이터 정규화
#         scaler = MinMaxScaler()
#         df[columns] = scaler.fit_transform(df[columns])

#         # 상태별로 데이터 분리
#         dem_data = df[df['target'] == 'Dem']
#         cn_data = df[df['target'] == 'CN']
#         mci_data = df[df['target'] == 'MCI']

#         # 레이더 차트 생성
#         fig = go.Figure()

#         fig.add_trace(go.Scatterpolar(
#             r=dem_data[columns].mean().values.tolist(),
#             theta=columns,
#             fill='toself',
#             name='Dem'
#         ))

#         fig.add_trace(go.Scatterpolar(
#             r=cn_data[columns].mean().values.tolist(),
#             theta=columns,
#             fill='toself',
#             name='CN'
#         ))

#         fig.add_trace(go.Scatterpolar(
#             r=mci_data[columns].mean().values.tolist(),
#             theta=columns,
#             fill='toself',
#             name='MCI'
#         ))

#         fig.update_layout(
#             polar=dict(
#                 radialaxis=dict(
#                     visible=True,
#                     range=[0, 1]
#                 )
#             ),
#             showlegend=True
#         )

#         # Streamlit에 그래프 표시
#         st.plotly_chart(fig)
#     with col2:
#         st.write("# 해당 레이더 차트를 보면 치매환자와 다른 클래스간의 차이가 나는것을 볼 수 있다..")
# if __name__ == '__main__':
#     main()
   

# Data

# pca
def do_pca(df, components):
    pca = PCA(n_components=components)
    
    X = df.drop('target', axis=1)
    Y = df['target']
    
    pca_df = pd.DataFrame(pca.fit_transform(X))
    pca_df['target'] = Y
    
    return pca_df




def main():

    
    # Streamlit 애플리케이션의 제목을 설정합니다.
    st.title('차원 축소')
    
    # 컬럼 레이아웃 생성
    col1, col2 = st.columns([1, 1])  # 첫 번째 컬럼은 두 번째 컬럼보다 2배 너비로 설정합니다.
    # col3 = st.columns()  # 첫 번째 컬럼은 두 번째 컬럼보다 2배 너비로 설정합니다.

    # 2차원 산점도를 생성하고 첫 번째 컬럼에 표시합니다.
    with col1:
        st.subheader('2D Scatter Plot')
        pca_df_2d = do_pca(df, 2)
        fig_2d = px.scatter(pca_df_2d, x=0, y=1, color='target', symbol='target',
                            title='PCA Plot (2 Components)', labels={'0': 'PC1', '1': 'PC2'},
                            template='plotly_dark', hover_data=['target'])
        fig_2d.update_traces(marker=dict(size=6))
        st.plotly_chart(fig_2d)
    
    # 3차원 산점도를 생성하고 두 번째 컬럼에 표시합니다.
    with col2:
        st.subheader('3D Scatter Plot')
        pca_df_3d = do_pca(df, 3)
        fig_3d = px.scatter_3d(pca_df_3d, x=0, y=1, z=2, color='target', symbol='target',
                            title='PCA Plot (3 Components)', labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'},
                            template='plotly_dark', hover_data=['target'])
        fig_3d.update_traces(marker=dict(size=2))
        
        fig_3d.update_layout(
            scene=dict(
                aspectmode='manual',  # 직육면체 비율을 수동으로 조정
                aspectratio=dict(x=2, y=1, z=1)  # x축 비율을 2로 조정하여 더 길게 표현
            )
        )
    
        st.plotly_chart(fig_3d)

    st.write("1. 비 치매군과 경도 인지 장애 간 데이터가 패턴이 유사하게 보이는 것만 같습니다.")
    st.write("가끔 두 클래스끼리 밀집된 지역이 있기도 하지만 대부분의 영역을 살펴 보았을 때 각 클래스 간 밀집되어 있는 모습을 볼 수 있습니다.")
    st.write("2. 치매군은 이상치처럼 멀리 나와있는 경우가 많습니다. 중앙에 분포된 두 개의 치매 군을 제외하고 말입니다.")
if __name__ == '__main__':
    main()

# GitHub의 이미지 URL
github_image_url = 'https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/full_pca.gif'

st.image(github_image_url, width=500)


df_test = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/test_f1_score.csv')
df_valid = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/valid_f1_score.csv')
df_test = df_test.sort_values(by = 'f1_score', ascending = False)
df_valid = df_valid.sort_values(by = 'f1_score', ascending = False)


# f1 score
def main():
    col5 , col6 = st.columns([1,1])
    with col5 :
        # Create the bar graph
        fig = px.bar(df_test, x='f1_score', y='model', orientation='h', color='model',
                labels={'f1_score': 'F1 Score', 'model': 'Model'})

    # Update layout
        fig.update_layout(title='Test set F1 Scores for Different Models',
                    xaxis=dict(title='f1_score'),
                    yaxis=dict(title='Model'))

        st.plotly_chart(fig)
    with col6:
        st.write('이것은 Test F1입니다.')

if __name__ == '__main__':
    main()


def main():
    col7, col8 = st.columns([1,1])
    with col7:
        # Create the bar graph
        fig = px.bar(df_valid, x='f1_score', y='model', orientation='h', color='model',
                labels={'f1_score': 'F1 Score', 'model': 'Model'})

    # Update layout
        fig.update_layout(title='Validation set F1 Scores for Different Models',
                    xaxis=dict(title='f1_score'),
                    yaxis=dict(title='Model'))

        st.plotly_chart(fig)
    with col8:
        st.write('이것은 Valid F1입니다.')
   
if __name__ == '__main__':
    main()


test_roc = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/TEST_ROC.csv')
valid_roc = pd.read_csv('https://raw.githubusercontent.com/yonghyeun/Dementia_Analysis/main/data/Modeling/VALID_ROC.csv')



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

def main():
    col9,col10 = st.columns([1,1])
    with col9:
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
    with col10:
        st.write('이것은 Test ROC입니다.')
    col11, col12 = st.columns([1,1])
    with col11:
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
    with col12:
        st.write('이것은 Valid ROC입니다.')
if __name__ == '__main__':
    main()


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
