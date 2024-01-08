import numpy as np
import pandas as pd

import json
# 从 JSON 文件读取数据到字典
with open('station_info.json', 'r') as json_file:
    station_info = json.load(json_file)

# Feature engineering
def get_featured_data(df, station_info):
    def get_flow(df):
        features = ['STATION_ID', 'TIME', 'TAKE_BIKES']
        data = pd.DataFrame(index=range(df.shape[0] - 1),columns=features)
    
        for i in range(1, df.shape[0]):
            data.iloc[i-1]['STATION_ID'] = df.iloc[i-1]["STATION_ID"]
            data.iloc[i-1]['TIME'] = df.iloc[i-1]["TIME"]
        
            y1 = df.iloc[i-1]["AVAILABLE_BIKE_STANDS"] - df.iloc[i]["AVAILABLE_BIKE_STANDS"]
            y2 = df.iloc[i-1]["AVAILABLE_BIKES"] - df.iloc[i]["AVAILABLE_BIKES"]
    
            # data.iloc[i-1]['BRING_BIKE_STANDS'] = y1
            # data.iloc[i-1]['TAKE_USING'] = y2 + y1
            
            data.iloc[i-1]['TAKE_BIKES'] = y2 + y2 + y1
        return data

    
    for idx, id in enumerate(station_info):
        if idx == 0:
            y = get_flow(df[df["STATION_ID"] == int(id)])
        else:
            y = pd.concat([y, get_flow(df[df["STATION_ID"] == int(id)])], axis = 0)
    y.index = range(y.shape[0])
    return y


def get_trainable_data(df, isolate_station, period):
    # Get the start-end TimeStamps
    if period == 'pre':
        start = pd.to_datetime("01-08-2018", format='%d-%m-%Y')
        end = pd.to_datetime("13-03-2020", format='%d-%m-%Y')
    elif period == 'dur':
        start = pd.to_datetime("13-03-2020", format='%d-%m-%Y')
        end = pd.to_datetime("27-01-2022", format='%d-%m-%Y')
    elif period == 'post':
        start = pd.to_datetime("27-01-2022", format='%d-%m-%Y')
        end = pd.to_datetime("25-12-2023", format='%d-%m-%Y')
    elif period == 'cross_validation':
        start = pd.to_datetime("01-08-2018", format='%d-%m-%Y')
        end = pd.to_datetime("31-10-2020", format='%d-%m-%Y')

    
    # Get full time index
    t_full = pd.array(pd.DatetimeIndex(df.iloc[:,1]).astype(np.int64)) / 1e9
    t_start = pd.DataFrame([start]).astype(np.int64) / 1e9
    t_end = pd.DataFrame([end]).astype(np.int64) / 1e9
    
    t = np.extract([np.asarray(t_full >= t_start[0][0]) &  np.asarray(t_full <= t_end[0][0])], t_full)
    # t.shape
    
    # Get the STATION_ID 
    id = np.extract([np.asarray((t_full>=t_start[0][0])) & np.asarray((t_full<=t_end[0][0]))], df.iloc[:,0]) 

    
    # Get the sampling time period
    dt = t[id == 2][1] - t[id == 2][0]
    # print("Data sampling interval is %d secs." %dt)

    t = (t - t[0]) / 60 / 60 / 24 # convert timestamp to days
    
    y = np.extract([np.asarray((t_full>=t_start[0][0])) & np.asarray((t_full<=t_end[0][0]))], df.iloc[:,2]).astype(np.float64)
    # y.shape
    y = (y - y.mean())/y.std()
    
    if isolate_station:
        y_2d = []
        t_2d = []
        for i in station_info:
            y_2d.append(y[id == int(i)])
            t_2d.append(t[id == int(i)])
        y_2d = np.array(y_2d)
        t_2d = np.array(t_2d)
        return y_2d, t_2d, id, dt
    else:
        return y, t, id, dt


def main(period, isolate_station):
    # isolate_station = False for 1-dim y, True for 2-dim y
    if period == 'pre':
        df_pre = pd.read_hdf('pre.h5', 'df')
        df = get_featured_data(df_pre, station_info)
    elif period == 'dur':
        df_dur = pd.read_hdf('dur.h5', 'df')
        df = get_featured_data(df_dur, station_info)
    elif period == 'post':
        df_post = pd.read_hdf('post.h5', 'df')
        df = get_featured_data(df_post, station_info)
    elif period == 'cross_validation':
        df_pre = pd.read_hdf('pre.h5', 'df')
        df = get_featured_data(df_pre, station_info)
        
    y, t, id, dt = get_trainable_data(df, isolate_station, period)
    
    return y, t, id, dt, station_info