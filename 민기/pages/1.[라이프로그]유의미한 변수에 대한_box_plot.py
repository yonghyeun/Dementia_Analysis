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
#     st.plotly_chart(fig)

# def main():
#     st.title('Box Plots')

#     # Assuming you have the DataFrame 'df' containing the data

#     # Create buttons for each target column
#     buttons = []
#     for col in target_columns:
#         button = st.button(col)
#         buttons.append(button)

#     # Create 7x2 grid layout
#     columns = st.columns(2)
#     for i, col in enumerate(target_columns):
#         if buttons[i]:
#             with columns[i % 2]:
#                 create_boxplot(col)

# if __name__ == '__main__':
#     main()
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
#     st.plotly_chart(fig)

# def main():
#     st.title('Box Plots')

#     # Assuming you have the DataFrame 'df' containing the data

#     # Create buttons for each target column in a horizontal layout
#     button_cols = st.columns(len(target_columns))
#     buttons = []
#     for i, col in enumerate(target_columns):
#         button = button_cols[i].button(col, key=col)
#         buttons.append(button)

#     # Create 7x2 grid layout
#     columns = st.columns(2)
#     for i, col in enumerate(target_columns):
#         if buttons[i]:
#             with columns[i % 2]:
#                 create_boxplot(col)

# if __name__ == '__main__':
#     main()


# 버튼으로 불러오는 너낌
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
#     st.plotly_chart(fig)

# def main():
#     st.title('Box Plots')

#     # Assuming you have the DataFrame 'df' containing the data

#     # Create a 2x7 grid layout for buttons
#     button_rows = st.columns(7)
#     buttons = []
#     # selected_plots = []
#     for i, col in enumerate(target_columns):
#         button = button_rows[i // 2].button(col, key=col)
#         buttons.append(button)
#     #     if button:
#     #         selected_plots.append(create_boxplot(col))
#     # if len(selected_plots) > 0:
#     #     st.plotly_chart(px.subplots(selected_plots, shared_yaxes=True))

#     # Create 7x2 grid layout for box plots
#     columns = st.columns(7)
#     for i, col in enumerate(target_columns):
#         if buttons[i]:
#             with columns[i % 7]:
#                 create_boxplot(col)
#     # if len(selected_plots) > 0:
#     #     st.plotly_chart(px.make_subplots(selected_plots, shared_yaxes=True))

# if __name__ == '__main__':
#     main()

# 버튼 사용한거
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
#     st.plotly_chart(fig)

# def main():
#     st.title('Box Plots')

#     # Assuming you have the DataFrame 'df' containing the data

#     # Create a 2x7 grid layout for buttons
#     button_rows = st.columns(7)
#     selected_buttons = []
#     for i, col in enumerate(target_columns):
#         button = button_rows[i // 2].button(col, key=col)
#         if button:
#             selected_buttons.append(col)

#     # Display the selected charts in a 1-column layout
#     for button in selected_buttons:
#         create_boxplot(button)

# if __name__ == '__main__':
#     main()

# 멀티 셀렉트 사용한거
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
# # 사이드바 이용
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
#     st.plotly_chart(fig)

# def main():
#     st.title('Box Plots')

#     # Assuming you have the DataFrame 'df' containing the data

#     # Create multiselect for selecting charts
#     selected_charts = st.sidebar.multiselect("Select Charts", target_columns)

#     # Create box plots for selected charts
#     for chart in selected_charts:
#         create_boxplot(chart)

# if __name__ == '__main__':
#     main()

