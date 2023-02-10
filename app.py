import streamlit as st # web development
import numpy as np # np mean, np random
import pandas as pd # read csv, df manipulation
import time # to simulate a real time data, time loop
import plotly.express as px # interactive charts
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

st.set_page_config(
    page_title = 'Road Volumes Estimation',
    page_icon = '✅',
    layout = 'wide',
)
st.markdown("<h1 style='text-align: center; color: black;background-color:#F5F5F5'>Road Volumes Estimation</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: justify; color: gray;'>Vehicular traffic counting is a system that allows counting and recording the number of vehicles that pass through a specific location in a specific period of time. This system can be implemented using computer vision technologies, such as YOLOv8 and ByteTrack. YOLOv8 is an object detection neural network that uses a deep learning architecture to detect and classify objects in an image or video.</h4>", unsafe_allow_html=True)
st.markdown('')
st.markdown("<h4 style='text-align: justify; color: gray;'>ByteTrack, on the other hand, is a video object tracking tool that allows tracking specific objects throughout a video and recording their position over time. In the context of vehicular traffic counting, ByteTrack can be used to track individual vehicles throughout a video and record their position over time, allowing the counting of the total number of vehicles that have passed.</h4>", unsafe_allow_html=True)
st.markdown('')
st.markdown("<h4 style='text-align: justify; color: gray;'>In summary, vehicular traffic counting using YOLOv8 and ByteTrack involves using computer vision technologies to detect and track vehicles in a video and then count the total number of vehicles that have passed through a location in a specific period of time.</h4>", unsafe_allow_html=True)           
st.markdown('')
st.markdown("<h6 style='text-align: justify; color: gray;'>Text created using ChatGPT.</h6>", unsafe_allow_html=True) 

df_train = pd.read_csv('train.csv')
df = pd.read_csv("df.csv")
df_trajectories = pd.read_csv("df_scatter.csv")
df_trajectories['line_name'] = df_trajectories['line_name'].astype('str')
df_trajectories['time_video'] = df_trajectories['time_video'] / 60

df_lines = pd.read_csv("df_lines.csv")
df_lines['time_video'] = df_lines['time_video'] / 60

df_lines_ac = pd.read_csv("df_lines_ac.csv")
df_lines_ac['time_video'] = df_lines_ac['time_video'] / 60

df_gate_total = pd.read_csv("df_gate_total.csv")
df_gate_total['line_name'] = df_gate_total['line_name'].astype('str')
df_class_total = pd.read_csv("df_class_total.csv")
 #filters 


placeholder = st.empty()

# while True:
with placeholder.container():
    
    # df_gate_total = df_gate_total[(df.line_name.isin(selected_movement))]
    # df_class_total = df_class_total[(df_class_total.nomClass.isin(selected_clase))]
    # st.dataframe(df_class_total)

  
    # st.dataframe(df)
    
    # st.write('chile2.gif')
    file_ = open("chile2.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()
    demo = st.expander('Demo')
    with demo:
        with st.container():
            st.markdown('')
            st.markdown("<h1 style='text-align: center; color: black;background-color:#F5F5F5'>Demo</h1>", unsafe_allow_html=True)
            st.markdown("<h4 style='text-align: center; color: gray;'>Traffic counting in a road street intersection in Providencia, Santiago de Chile.</h4>", unsafe_allow_html=True)  
            st.markdown(
                f'<center><img src="data:image/gif;base64,{data_url}" width=100%></center>',
                unsafe_allow_html=True,
            )

    st.markdown('')
    st.markdown("<h1 style='text-align: center; color: black;background-color:#F5F5F5'>Training Yolov8 Results</h1>", unsafe_allow_html=True)

    exp = st.expander('Show Training results')
    with st.container():
        with exp: 
            fig_1, fig_2 = st.columns(2)
            with fig_1:
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
                fig.update_layout(autosize=True)
                st.plotly_chart(fig, use_container_width=True)

        # fig_train2 = st.expander('Metrics')
            with fig_2:    
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
                fig2.update_layout(autosize=True)
                st.plotly_chart(fig2, use_container_width=True)
        

    
    #bar charts
    st.markdown('')
    st.markdown("<h1 style='text-align: center; color: black;background-color:#F5F5F5'>Bar Charts / Volumes Estimation</h1>", unsafe_allow_html=True)
    fig_col1, fig_col2 = st.columns(2)
    with st.container():
        with fig_col1:
            st.markdown("### Total by Gates")
            fig = px.bar(df_gate_total, x='line_name', y='Total',color='line_name',text_auto=True)
            fig.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,font=dict(size= 20)
                    
                ))
            st.plotly_chart(fig, use_container_width=True)
        # st.dataframe(df_gate_total)
        with fig_col2:
            st.markdown("### Total by Class")
            fig2 = px.bar(df_class_total, x='nomClass', y='Total',color='nomClass',text_auto=True)
            fig2.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,font=dict(size= 20)
                ))
            st.plotly_chart(fig2, use_container_width=True)
        st.markdown('')
        st.markdown("<h4 style='text-align: justify; color: gray;'>The bar chart displays the results of a traffic count conducted using the YOLOv8 app. The chart presents the number of vehicles detected in a specific area, grouped by type of vehicle. Each bar in the chart corresponds to a different type of vehicle, such as cars, trucks, buses, motorcycles, and bicycles, and the height of each bar represents the number of vehicles detected of that type. The Y-axis is labeled with the number of vehicles, while the X-axis is labeled with the different types of vehicles. The bars are color-coded to distinguish between the different types of vehicles. This bar chart provides a visual representation of the traffic data collected by the YOLOv8 app, allowing for easy analysis and understanding of the vehicle composition in the area.</h4>", unsafe_allow_html=True)  
        st.markdown("<h6 style='text-align: justify; color: gray;'>Text created using ChatGPT.</h6>", unsafe_allow_html=True) 

    #Time series and trajectories with filters
    st.markdown('')
    st.markdown("<h1 style='text-align: center; color: black;background-color:#F5F5F5'>Filters for time series and trajectories charts</h1>", unsafe_allow_html=True)
    selected_movement = st.multiselect("Gate", df.line_name.unique(),df.line_name.unique())
    selected_clase = st.multiselect("class", df.nomClass.unique(),df.nomClass.unique())
    df = df[(df.line_name.isin(selected_movement)) & (df.nomClass.isin(selected_clase))]
    df_trajectories = df_trajectories[(df.line_name.isin(selected_movement)) & (df_trajectories.nomClass.isin(selected_clase))]
    df_lines = df_lines[(df.line_name.isin(selected_movement)) & (df_lines.nomClass.isin(selected_clase))]
    df_lines_ac = df_lines_ac[(df.line_name.isin(selected_movement)) & (df_lines_ac.nomClass.isin(selected_clase))]
    #lines charts
    st.markdown('')
    st.markdown("<h1 style='text-align: center; color: black;background-color:#F5F5F5'>Time Series Charts</h1>", unsafe_allow_html=True)
    fig_time, fig_time_ac = st.columns(2)
    with st.container():
        with fig_time:
            st.markdown("### Total Vehicles")
            fig_time_0 = px.line(df_lines, x='time_video', y='c', color='nomClass', markers=True)
            fig_time_0.update_layout(autosize=True)
            fig_time_0.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,font=dict(size= 20)
                ))
            st.plotly_chart(fig_time_0, use_container_width=True)
            st.markdown('')
            st.markdown("<h4 style='text-align: justify; color: gray;'>A stationary chart in vehicular traffic is a type of graphical representation that shows the number of vehicles passing through a specific point over a given period of time. A stationary chart is one in which the number of vehicles passing through a specific point does not change over time, meaning the flow rate of vehicles is constant. These charts are used to analyze vehicular traffic and to identify patterns and trends in traffic, such as the daily, weekly, or monthly distribution of vehicles. They can also be used to evaluate the effectiveness of traffic control measures and to plan the construction of roads and the placement of traffic lights.</h4>", unsafe_allow_html=True)  
            st.markdown("<h6 style='text-align: justify; color: gray;'>Text created using ChatGPT.</h6>", unsafe_allow_html=True) 

            # st.dataframe(df_lines, width=2000, height=None)
        with fig_time_ac:
            st.markdown("### Accumulated")
            fig_time_acc = px.line(df_lines_ac, x='time_video', y='GCS', color='nomClass', markers=True)
            fig_time_acc.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,font=dict(size= 20)
                ))
            st.plotly_chart(fig_time_acc, use_container_width=True)
            # st.write(fig_time_acc)
            st.markdown('')
            st.markdown("<h4 style='text-align: justify; color: gray;'>A cumulative flow chart in vehicle traffic counts is a type of graphical representation that shows the total number of vehicles that have passed through a specific point over a given period of time. Unlike the stationary chart, which shows the number of vehicles passing through a point at a given moment, the cumulative flow chart shows the total number of vehicles that have passed through a point over time.The cumulative flow chart is useful for analyzing the evolution of vehicle traffic at an intersection over time. For example, it can be used to identify patterns and trends in traffic, such as daily, weekly, or monthly vehicle distributions. It can also be used to evaluate the effectiveness of traffic control measures and to plan the construction of roads and the location of traffic lights.</h4>", unsafe_allow_html=True)  
            st.markdown("<h6 style='text-align: justify; color: gray;'>Text created using ChatGPT.</h6>", unsafe_allow_html=True) 

        # st.dataframe(df_lines_ac, width=2000, height=None)

    # st.markdown("### DataFrame")
    # st.dataframe(df, width=2000, height=None)

    # st.dataframe(df_class_total)

    #trajectories
    st.markdown('')
    st.markdown("<h1 style='text-align: center; color: black;background-color:#F5F5F5'>Trajectories of detected objects</h1>", unsafe_allow_html=True)
    fig_col8, fig_col9 = st.columns(2)
    with st.container():
        with fig_col8:
            st.markdown("### Trajectories by Gates")
            fig8 = px.scatter(df_trajectories, x='xc',y='yc', color='line_name')
            fig8.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,font=dict(size= 20)
                ))
            st.plotly_chart(fig8, use_container_width=True)
        with fig_col9:
            st.markdown("### Trajectories by Class")
            fig9 = px.scatter(df_trajectories, x='xc',y='yc', color='nomClass')
            fig9.update_layout(legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,font=dict(size= 20)
                ))
            st.plotly_chart(fig9, use_container_width=True)



        
    st.markdown('')

    st.markdown("<h4 style='text-align: justify; color: gray;'>The trajectory of a vehicle is the path it takes through the scene, including its speed, direction, and position.The program would likely save the trajectories of each vehicle in a data structure, such as an array or a database, for later analysis. The saved trajectories could then be used to generate various statistics about the traffic flow, such as the average speed of vehicles, the number of vehicles that passed through the scene, and the direction of travel..</h4>", unsafe_allow_html=True)  
    st.markdown('')
    st.markdown("<h3 style='text-align: center; color: black;background-color:#F5F5F5'>References</h3>", unsafe_allow_html=True)
    st.markdown("<h6 style='text-align: justify; color: gray;'>ChatGPT. (2023). Definitions of Road Volumes Estimations in Vehicular Traffic retrieved from OpenAI: https://openai.com/ .</h6>", unsafe_allow_html=True) 

    # st.markdown("### Detailed Data View")
    # st.dataframe(df_trajectories)
    time.sleep(1)
#placeholder.empty()












