import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime

# This the file loading all data and return them in 3 type classes (before during after) of pademic period
def generate_filenames(start_year, end_year, pattern):
    filenames = []
    if pattern == 0:
        # Monthly data filenames
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                filename = f"dublinbike-historical-data-{year}-{month:02d}.csv"
                filenames.append(filename)
    elif pattern == 1:
        # Quarterly data filenames
        quarters = [(1, 4), (4, 7), (7, 10), (10, 1)]
        for year in range(start_year, end_year):
            for start_month, end_month in quarters:
                start_date = f"{year}{start_month:02d}01"
                end_year_shift = year if end_month != 1 else year + 1
                end_date = f"{end_year_shift}{end_month:02d}01"
                filename = f"dublinbikes_{start_date}_{end_date}.csv"
                filenames.append(filename)

    return filenames


def read_data():
    start_year = 2018
    end_year = 2023
    df = pd.DataFrame()
    # Choose 0 for monthly data, 1 for quarterly data
    for pattern in range(2):
        file_list = generate_filenames(start_year, end_year, pattern)
        for file in file_list: # Loading data
            file_name = "datafiles/" + file
            if os.path.exists(file_name): 
                print(f"Load file: {file_name}")
                data = pd.read_csv(file_name)

                # Unify the colunm indexes
                data.rename(columns = {"AVAILABLE BIKE STANDS": "AVAILABLE_BIKE_STANDS"}, inplace=True)
                data.rename(columns = {"AVAILABLE BIKES": "AVAILABLE_BIKES"}, inplace=True)
                data.rename(columns = {"BIKE STANDS": "BIKE_STANDS"}, inplace=True)
                
                df = pd.concat([df, data], axis=0, ignore_index=True)
    
    station_info_dict = {}
    for id in df["STATION ID"]:
        if id in station_info_dict:
            continue
        else:
            station_info_dict[id] = {
                "NAME": df[df["STATION ID"] == id]["NAME"].values[0],
                "ADDRESS": df[df["STATION ID"] == id]["ADDRESS"].values[0],
                "LATITUDE": df[df["STATION ID"] == id]["LATITUDE"].values[0],
                "LONGITUDE": df[df["STATION ID"] == id]["LONGITUDE"].values[0]
            }
    station_info_dict = dict(sorted(station_info_dict.items(), key = lambda item: item[0]))
    
    df["TIME"] = pd.to_datetime(df["TIME"])
    # Split March 2020 into pre and post pandemic
    # March 13th, the first day schools were closed 
    # (https://www.irishtimes.com/health/2023/05/05/covid-emergency-is-over-20-key-moments-of-pandemic-that-changed-the-world/)
    timepoint_begin = pd.Timestamp('2020-03-13 00:00:00')
    
    # Split January 2022 into pre and post pandemic
    # Jan 28th 2022, the day the HSE stopped releasing COVID-19 figures 
    # (https://www.irishtimes.com/health/2023/05/05/covid-emergency-is-over-20-key-moments-of-pandemic-that-changed-the-world/)
    timepoint_end = pd.Timestamp('2022-01-27 23:59:59')

    df = df.drop_duplicates()
    df_pre = df[df["TIME"] < timepoint_begin]
    df_dur = df[(df["TIME"] >= timepoint_begin) & (df["TIME"] <= timepoint_end)]
    df_post = df[df["TIME"] > timepoint_end]
    
    return df_pre, df_dur, df_post, station_info_dict

# Round a time string to the nearest 5 minutes in a datetime object
def round_to_5_minutes(time_stamp: pd.Timestamp) -> datetime:
    # Parse the input string into a datetime object
    dt = time_stamp.to_pydatetime()

    # Round down to the nearest 5 minutes
    rounded_dt = datetime(dt.year, dt.month, dt.day, dt.hour, (dt.minute // 5) * 5)

    return rounded_dt

# Round a time string to the nearest 8 hour in a datetime object
def round_to_hour(time_stamp: pd.Timestamp) -> datetime:
    # Parse the input string into a datetime object
    dt = time_stamp.to_pydatetime()

    # Round down to the nearest day
    rounded_dt = datetime(dt.year, dt.month, dt.day, (dt.hour // 8) * 8)

    return rounded_dt
    
# Round a time string to the nearest a day in a datetime object
def round_to_day(time_stamp: pd.Timestamp) -> datetime:
    # Parse the input string into a datetime object
    dt = time_stamp.to_pydatetime()

    # Round down to the nearest day
    rounded_dt = datetime(dt.year, dt.month, dt.day)

    return rounded_dt
    
# Combine time/dates into one value per time
def combine_dates(df: pd.DataFrame, period: str) -> pd.DataFrame:
    # Round times to nearest 5 minutes
    # tqdm.pandas(desc=f"Rounding {period} times to closest 5 minutes.")
    # df['TIME'] = df['TIME'].progress_apply(round_to_5_minutes)

    # Round times to nearest a day
    # tqdm.pandas(desc=f"Rounding {period} times to closest day.")
    # df['TIME'] = df['TIME'].progress_apply(round_to_day)

    # # Round times to nearest an hour
    tqdm.pandas(desc=f"Rounding {period} times to closest hour.")
    df['TIME'] = df['TIME'].progress_apply(round_to_hour)

    # Aggregate times together, with all counts summed
    return df.groupby(df['TIME'], as_index=False).aggregate({'BIKE_STANDS': 'mean', 'AVAILABLE_BIKE_STANDS': 'mean', 'AVAILABLE_BIKES': 'mean'})

# Combine time/dates of each station into one value per time
def combine_dates_station(df: pd.DataFrame, info: dict, period: str) -> pd.DataFrame:
    df_new = pd.DataFrame()
    for station in info:
        temp = combine_dates(df[df["STATION ID"] == station], period)
        temp["STATION_ID"] = station
        df_new = pd.concat([df_new, temp], axis=0, ignore_index=True)
    return df_new


# Clean data, fill zeros to un-occured station
from itertools import product
def clean_data(df:pd.DataFrame, info: dict) -> pd.DataFrame:
    station_ids = list(info.keys()) # 从字典中提取站点ID

    # 步骤 1: 创建完整的时间序列
    min_time = df['TIME'].min()
    max_time = df['TIME'].max()
    all_times = pd.date_range(start=min_time, end=max_time, freq='8H') #D for day, H for hour, T for minute

    # 为每个站点创建完整的时间序列
    full_df = pd.DataFrame(list(product(station_ids, all_times)), columns=['STATION_ID', 'TIME'])

    # 合并数据集
    df = df.set_index(['STATION_ID', 'TIME'])
    full_df = full_df.set_index(['STATION_ID', 'TIME'])
    combined_df = full_df.join(df, how='left')

    # 填充缺失值
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.reset_index(inplace=True)
    return combined_df

import json
def main():
    df_pre, df_dur, df_post, station_info = read_data()
    df_pre, df_dur, df_post = combine_dates_station(df_pre, station_info, "pre-pandemic"), combine_dates_station(df_dur, station_info, "pandemic"), combine_dates_station(df_post, station_info, "post-pandemic")
    
    df_pre, df_dur, df_post = clean_data(df_pre, station_info), clean_data(df_dur, station_info), clean_data(df_post, station_info)  

    # 存储和读取 pandas DataFrame，
    df_pre.to_hdf("pre.h5", key='df', mode='w')
    df_dur.to_hdf("dur.h5", key='df', mode='w')
    df_post.to_hdf("post.h5", key='df', mode='w')

    # 将字典写入 JSON 文件
    with open('station_info.json', 'w') as json_file:
        json.dump(station_info, json_file)

if __name__ == "__main__":
    main()
