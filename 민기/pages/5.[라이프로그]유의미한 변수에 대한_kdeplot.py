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

activity_cols = ['activity_average_met', 'activity_cal_active', 'activity_cal_total',
       'activity_daily_movement', 'activity_high', 'activity_inactive',
       'activity_inactivity_alerts', 'activity_low', 'activity_medium',
       'activity_met_min_high', 'activity_met_min_inactive',
       'activity_met_min_low', 'activity_met_min_medium', 'activity_non_wear',
       'activity_rest', 'activity_score', 'activity_score_meet_daily_targets',
       'activity_score_move_every_hour', 'activity_score_recovery_time',
       'activity_score_stay_active', 'activity_score_training_frequency',
       'activity_score_training_volume', 'activity_steps', 'activity_total','target']

sleep_cols = ['sleep_awake', 'sleep_breath_average', 'sleep_deep', 'sleep_duration',
       'sleep_efficiency', 'sleep_hr_average', 'sleep_hr_lowest', 'sleep_light', 
       'sleep_midpoint_at_delta','sleep_midpoint_time', 'sleep_onset_latency', 'sleep_period_id',
       'sleep_rem', 'sleep_restless', 'sleep_rmssd', 'sleep_score',
       'sleep_score_alignment', 'sleep_score_deep', 'sleep_score_disturbances',
       'sleep_score_efficiency', 'sleep_score_latency', 'sleep_score_rem',
       'sleep_score_total', 'sleep_temperature_delta',
       'sleep_temperature_deviation', 'sleep_total','target']


def cross_tab(df):
    
    mean_ctab = round(df.groupby('target').mean(),2).astype(str)
    
    sd_ctab = round(df.groupby('target').std(),2).astype(str)
    
    result = mean_ctab + ' (±' + sd_ctab + ')'
    
    return result.T

activity_ctab = cross_tab(df[activity_cols])
sleep_ctab = cross_tab(df[sleep_cols])


def anova_test(df):
    
    selector = SelectKBest(score_func = f_classif)
    
    X = df.drop('target',axis = 1)
    Y = df['target']
    
    selector.fit(X,Y)
    
    result = pd.DataFrame({'F-statics':selector.scores_,
                            'p-value':selector.pvalues_}, index = selector.feature_names_in_).sort_values(by = 'p-value')
    
    
    return result 


activity_anova = anova_test(df[activity_cols])
sleep_anova = anova_test(df[sleep_cols])

def merge_ctab_anova(ctab,anova):
    
    result = pd.merge(ctab,anova,left_index = True, right_index = True).sort_values(by = 'p-value')
    
    result = result[['CN','MCI','Dem','F-statics','p-value']]
    
    return result

activity_result = merge_ctab_anova(activity_ctab,activity_anova)
sleep_result = merge_ctab_anova(sleep_ctab,sleep_anova)


activity_pvalue = activity_result[activity_result['p-value'] < 0.05]
sleep_pvalue = sleep_result[sleep_result['p-value'] < 0.05]

# 그래프 그리기
fig = go.Figure()
 
# 그래프 그리기
target_colors = {'CN':'#b2df8a','MCI':'#fdb462','Dem':'#fb6a4a'}
colors = ['green', 'orange', 'red']

p_value = activity_pvalue['p-value'].values
cols = activity_pvalue.index

for col in cols:
    fig = go.Figure()

    for target, color in target_colors.items():
        fig.add_trace(go.Histogram(
            x=df.loc[df['target'] == target, col],
            marker_color=color,
            opacity=0.7,
            name=target
        ))

    fig.add_trace(go.Scatter(
        x=[np.mean(df.loc[df['target'] == 'CN', col]), np.mean(df.loc[df['target'] == 'CN', col])],
        y=[0, 100],
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        showlegend=False
    ))

    fig.update_layout(
        title_text=col.replace('_', ' '),
        xaxis_title_text='',
        yaxis_title_text='Frequency',
        barmode='overlay',
        bargap=0.1,
        legend=dict(
            x=0.9,
            y=0.95,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        )
    )

    fig.add_annotation(
        x=np.mean(df.loc[df['target'] == 'CN', col]),
        y=80,
        text=f'P-value: {p_value[i]}',
        showarrow=False,
        font=dict(color='white'),
        bgcolor='black',
        opacity=0.8
    )

    # Streamlit에 그래프 출력
    st.plotly_chart(fig)


