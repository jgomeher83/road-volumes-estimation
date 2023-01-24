import streamlit as st # web development
import numpy as np # np mean, np random
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop
import plotly.express as px # interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title = 'Real-Time Data Science Dashboard',
    page_icon = '✅',
    layout = 'wide',
)
st.title("Real-Time / Live Data Science Dashboard")

df_train = pd.read_csv('train.csv')
df = pd.read_csv("df.csv")
df_trajectories = pd.read_csv("df_scatter.csv")
df_trajectories['line_name'] = df_trajectories['line_name'].astype('str')

df_lines = pd.read_csv("df_lines.csv")
df_lines['time_video'] = df_lines['time_video'] / 60

df_lines_ac = pd.read_csv("df_lines_ac.csv")
df_lines_ac['time_video'] = df_lines_ac['time_video'] / 60

df_gate_total = pd.read_csv("df_gate_total.csv")
df_gate_total['line_name'] = df_gate_total['line_name'].astype('str')
df_class_total = pd.read_csv("df_class_total.csv")
 #filters 
selected_movement = st.sidebar.multiselect("Gate", df.line_name.unique(),df.line_name.unique())
selected_clase = st.sidebar.multiselect("class", df.nomClass.unique(),df.nomClass.unique())

placeholder = st.empty()

# while True:
with placeholder.container():
    fig_train1, fig_train2, = st.columns(2)
    with fig_train1:     
        fig = make_subplots(rows=2, cols=3, subplot_titles=("train/box_loss", "train/cls_loss", "train/dfl_loss","val/box_loss","val/cls_loss","val/dfl_loss"))

        fig.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['train/box_loss']),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['train/cls_loss']),
                    row=1, col=2)
        fig.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['train/dfl_loss']),
                    row=1, col=3)               
        fig.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['val/box_loss']),
                    row=2, col=1)
        fig.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['val/cls_loss']),
                    row=2, col=2)
        fig.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['val/dfl_loss']),
                    row=2, col=3)
        fig.update_layout(showlegend=False, title_text="Training yolov8n results - Train/Validation Loss")
        st.write(fig)

    with fig_train2:     
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=("metrics/precision(B)", "metrics/recall(B)","metrics/mAP50(B)", "metrics/mAP50-95(B)"))
        fig2.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['metrics/precision(B)']),
                    row=1, col=1)
        fig2.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['metrics/recall(B)']),
                    row=1, col=2)
        fig2.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['metrics/mAP50(B)']),
                    row=2, col=1)
        fig2.add_trace(go.Scatter(x=df_train['epoch'], y=df_train['metrics/mAP50-95(B)']),
                    row=2, col=2)
        fig2.update_layout(showlegend=False, title_text="Training yolov8n results - Metrics")
        st.write(fig2)

    #lines charts
    fig_time, fig_time_ac = st.columns(2)
    with fig_time:
        st.markdown("### Total Vehicles")
        fig_time_0 = px.line(df_lines, x='time_video', y='c', color='nomClass')
        st.write(fig_time_0)
    with fig_time_ac:
        st.markdown("### Accumulated")
        fig_time_acc = px.line(df_lines_ac, x='time_video', y='GCS', color='nomClass')
        st.write(fig_time_acc)
    
    st.markdown("### DataFrame")
    st.dataframe(df, width=2000, height=None)

    #bar charts
    fig_col1, fig_col2 = st.columns(2)
    with fig_col1:
        st.markdown("### Total by Gates")
        fig = px.bar(df_gate_total, x='line_name', y='Total',color='line_name',text_auto=True)
        st.write(fig)
    # st.dataframe(df_gate_total)
    with fig_col2:
        st.markdown("### Total by Class")
        fig2 = px.bar(df_class_total, x='nomClass', y='Total',color='nomClass',text_auto=True)
        st.write(fig2)
    # st.dataframe(df_class_total)

    #trajectories
    fig_col8, fig_col9 = st.columns(2)
    with fig_col8:
        st.markdown("### Trajectories by Gates")
        fig8 = px.scatter(df_trajectories, x='xc',y='yc', color='line_name')
        st.write(fig8)
    with fig_col9:
        st.markdown("### Trajectories by Class")
        fig9 = px.scatter(df_trajectories, x='xc',y='yc', color='nomClass')
        st.write(fig9)

    # st.markdown("### Detailed Data View")
    # st.dataframe(df_trajectories)
    time.sleep(1)
#placeholder.empty()






