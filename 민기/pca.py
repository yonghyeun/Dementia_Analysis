import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import requests


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


import plotly.express as px

# def do_pca(df, components, make_plot=True):
#     pca = PCA(n_components=components)
    
#     X = df.drop('target', axis=1)
#     Y = df['target']
    
#     pca_df = pd.DataFrame(pca.fit_transform(X))
#     pca_df['target'] = Y
    
#     if make_plot:
#         target_colors = {'CN': 'green', 'MCI': 'orange', 'Dem': 'red'}
#         cmap = Y.map(target_colors)
        
#         if components == 2:
#             fig = px.scatter(pca_df, x=0, y=1, color='target', symbol='target',
#                              title='PCA Plot (2 Components)', labels={'0': 'PC1', '1': 'PC2'},
#                              template='plotly_dark', hover_data=['target'])
            
#             fig.update_traces(marker=dict(size=8))
#             fig.show()
        
#         if components == 3:
#             fig = px.scatter_3d(pca_df, x=0, y=1, z=2, color='target', symbol='target',
#                                 title='PCA Plot (3 Components)', labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'},
#                                 template='plotly_dark', hover_data=['target'])
            
#             fig.update_traces(marker=dict(size=8))
#             fig.show()
    
#     return pca_df

def do_pca(df, components):
    pca = PCA(n_components=components)
    
    X = df.drop('target', axis=1)
    Y = df['target']
    
    pca_df = pd.DataFrame(pca.fit_transform(X))
    pca_df['target'] = Y
    
    return pca_df



# def main():
#     # 데이터프레임을 생성합니다. 예시로 임의의 데이터를 사용합니다.
#     # Streamlit 애플리케이션의 제목을 설정합니다.
#     st.title('PCA Plot')
    
#     # 2차원 산점도를 생성하고 대시보드에 표시합니다.
#     st.subheader('2D Scatter Plot')
#     pca_df_2d = do_pca(df, 2)
#     fig_2d = px.scatter(pca_df_2d, x=0, y=1, color='target', symbol='target',
#                         title='PCA Plot (2 Components)', labels={'0': 'PC1', '1': 'PC2'},
#                         template='plotly_dark', hover_data=['target'])
#     fig_2d.update_traces(marker=dict(size=8))
#     st.plotly_chart(fig_2d)
    
#     # 3차원 산점도를 생성하고 대시보드에 표시합니다.
#     st.subheader('3D Scatter Plot')
#     pca_df_3d = do_pca(df, 3)
#     fig_3d = px.scatter_3d(pca_df_3d, x=0, y=1, z=2, color='target', symbol='target',
#                            title='PCA Plot (3 Components)', labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'},
#                            template='plotly_dark', hover_data=['target'])
#     fig_3d.update_traces(marker=dict(size=8))
#     st.plotly_chart(fig_3d)

# if __name__ == '__main__':
#     main()


# def main():
#     # 데이터프레임을 생성합니다. 예시로 임의의 데이터를 사용합니다.
 
#     # Streamlit 애플리케이션의 제목을 설정합니다.
#     st.title('PCA Plot')
    
#     # 컬럼 레이아웃 생성
#     col1, col2,col3= st.beta_columns(4)
    
#     # 2차원 산점도를 생성하고 첫 번째 컬럼에 표시합니다.
#     with col1:
#         st.subheader('2D Scatter Plot')
#         pca_df_2d = do_pca(df, 2)
#         fig_2d = px.scatter(pca_df_2d, x=0, y=1, color='target', symbol='target',
#                             title='PCA Plot (2 Components)', labels={'0': 'PC1', '1': 'PC2'},
#                             template='plotly_dark', hover_data=['target'])
#         fig_2d.update_traces(marker=dict(size=8))
#         st.plotly_chart(fig_2d)
    
#     # 3차원 산점도를 생성하고 두 번째 컬럼에 표시합니다.
#     with col2:
#         st.subheader('3D Scatter Plot')
#         pca_df_3d = do_pca(df, 3)
#         fig_3d = px.scatter_3d(pca_df_3d, x=0, y=1, z=2, color='target', symbol='target',
#                                title='PCA Plot (3 Components)', labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'},
#                                template='plotly_dark', hover_data=['target'])
#         fig_3d.update_traces(marker=dict(size=8))
#         st.plotly_chart(fig_3d)
#     with col3:
#         st.subheader('2D Scatter Plot')
#         pca_df_2d = do_pca(df, 2)
#         fig_2d = px.scatter(pca_df_2d, x=0, y=1, color='target', symbol='target',
#                             title='PCA Plot (2 Components)', labels={'0': 'PC1', '1': 'PC2'},
#                             template='plotly_dark', hover_data=['target'])
#         fig_2d.update_traces(marker=dict(size=8))
#         st.plotly_chart(fig_2d)
    


# if __name__ == '__main__':
#     main()


# import streamlit as st
# import pandas as pd
# import plotly.express as px

def main():

    
    # Streamlit 애플리케이션의 제목을 설정합니다.
    st.title('PCA Plot')
    
    # 컬럼 레이아웃 생성
    col1, col2 = st.columns([2, 1,])  # 첫 번째 컬럼은 두 번째 컬럼보다 2배 너비로 설정합니다.
    
    # 2차원 산점도를 생성하고 첫 번째 컬럼에 표시합니다.
    with col1:
        st.subheader('2D Scatter Plot')
        pca_df_2d = do_pca(df, 2)
        fig_2d = px.scatter(pca_df_2d, x=0, y=1, color='target', symbol='target',
                            title='PCA Plot (2 Components)', labels={'0': 'PC1', '1': 'PC2'},
                            template='plotly_dark', hover_data=['target'])
        fig_2d.update_traces(marker=dict(size=8))
        st.plotly_chart(fig_2d)
    
    # 3차원 산점도를 생성하고 두 번째 컬럼에 표시합니다.
    with col2:
        st.subheader('3D Scatter Plot')
        pca_df_3d = do_pca(df, 3)
        fig_3d = px.scatter_3d(pca_df_3d, x=0, y=1, z=2, color='target', symbol='target',
                               title='PCA Plot (3 Components)', labels={'0': 'PC1', '1': 'PC2', '2': 'PC3'},
                               template='plotly_dark', hover_data=['target'])
        fig_3d.update_traces(marker=dict(size=8))
        st.plotly_chart(fig_3d)

if __name__ == '__main__':
    main()

import requests

# GitHub의 이미지 URL
github_image_url = 'https://raw.githubusercontent.com/username/repository/master/image.jpg'

# 이미지 다운로드 함수
def download_image(url):
    response = requests.get(url)
    return response.content

# Streamlit 앱
def main():
    st.title('GitHub 이미지 보기')
    
    # 이미지 다운로드 및 표시
    image_content = download_image(github_image_url)
    st.image(image_content)

if __name__ == '__main__':
    main()
