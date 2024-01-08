import tensorflow as tf
import tensorflow.keras as keras

import pandas as pd
import numpy as np
import sys, math
import matplotlib.pyplot as plt

import preprocessing

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True

def feature_all_time_series(q = 3, lag = 3):
    def feature_time_series(q, lag, plot, y, t, id, dt):
    # q-step ahead prediction
        stride = 1

        # m = math.floor(30*7*24*60*60 / dt) # number of samples per month
        w = math.floor(7*24*60*60 / dt) # number of samples per week
        d = math.floor(24*60*60 / dt)

        len = y.size - w - lag * w - q

        XX = y[q: q+len: stride]
            
        for i in range(1, lag):
            temp = y[i*w+q: i*w+q+len: stride]
            XX = np.column_stack((XX, temp))

        for i in range(0, lag):
            temp = y[i*d+q: i*d+q+len: stride]
            XX = np.column_stack((XX, temp))

        for i in range(0, lag):
            temp = y[i+q: i+q+len: stride]
            XX = np.column_stack((XX, temp))

        yy = y[lag*w+w+q: lag*w+w+q+len: stride]
        tt = t[lag*w+w+q: lag*w+w+q+len: stride]
        iidd = id[lag*w+w+q: lag*w+w+q+len: stride]
        return XX, yy, tt, iidd

    for idx, idd in enumerate(station_info):
        idd = int(idd)
        if idd == 1:
            X, Y, T, ID = feature_time_series(q, lag, True, y[id==idd], t[id==idd], id[id==idd], dt)
        else:
            X0, y0, t0, id0 = feature_time_series(q, lag, True, y[id==idd], t[id==idd], id[id==idd], dt)
            X = np.row_stack((X, X0))
            Y = np.concatenate((Y, y0))
            T = np.concatenate((T, t0))
            ID = np.concatenate((ID, id0))
    X.shape, Y.shape, T.shape, ID.shape
    return X, Y, T, ID



from sklearn.utils import resample

# 假设 model 是你的TensorFlow模型
# data, labels 是你的测试数据和标签

def bootstrap_evaluate(model, data, labels, n_iterations=100, sample_size=0.2):
    scores = list()
    n_size = int(len(data) * sample_size)
    
    for i in range(n_iterations):
        # 准备Bootstrap样本
        indices = np.random.randint(0, len(data), n_size)
        sample_data, sample_labels = data[indices], labels[indices]
        
        # 评估模型
        loss, accuracy = model.evaluate(sample_data, sample_labels, verbose=0)
        scores.append(accuracy)
    
    # 分析Bootstrap结果
    mean_score = np.mean(scores)
    std_dev = np.std(scores)
    return scores


def main():
    model = tf.keras.models.load_model('model/LSTM')
    
    
    isolate_station = False
    y, t, id, dt, station_info = preprocessing.main('pre', isolate_station)
    y_pre = y; t_pre = t; id_pre = id; dt_pre = dt
    X_pre, Y_pre, T_pre, ID_pre = feature_all_time_series(q = 3, lag = 3)
    # 在训练之前，确保 x_train 和 x_test 的形状是 (样本数, 1, 特征数)
    X_LSTM = X_pre.reshape((X_pre.shape[0], 1, 9))
    X_LSTM.shape
    
    
    
    isolate_station = False
    y, t, id, dt, station_info = preprocessing.main('dur', isolate_station)
    y_dur = y; t_dur = t; id_dur = id; dt_dur = dt
    X_dur, Y_dur, T_dur, ID_dur = feature_all_time_series(q = 3, lag = 3)
    # 在训练之前，确保 x_train 和 x_test 的形状是 (样本数, 1, 特征数)
    X_LSTM_dur = X_dur.reshape((X_dur.shape[0], 1, 9))
    X_LSTM_dur.shape
    
    
    
    isolate_station = False
    y, t, id, dt, station_info = preprocessing.main('post', isolate_station)
    y_post = y; t_post = t; id_post = id; dt_post = dt
    X_post, Y_post, T_post, ID_post = feature_all_time_series(q = 3, lag = 3)
    # 在训练之前，确保 x_train 和 x_test 的形状是 (样本数, 1, 特征数)
    X_LSTM_post = X_post.reshape((X_post.shape[0], 1, 9))
    X_LSTM_post.shape
    
    
    model.evaluate(X_LSTM, Y_pre)
    scores_pre = bootstrap_evaluate(model, X_LSTM, Y_pre)
    
    model.evaluate(X_LSTM_dur, Y_dur)
    scores_dur = bootstrap_evaluate(model, X_LSTM_dur, Y_dur)
    
    model.evaluate(X_LSTM_post, Y_post)
    scores_post = bootstrap_evaluate(model, X_LSTM_post, Y_post)
    
    print("Mean and std of pre-pandemic are:")
    print(np.mean(scores_pre),np.std(scores_pre))
    
    print("Mean and std of pandemic are:")
    print(np.mean(scores_dur), np.std(scores_dur))
    
    print("Mean and std of post-pandemic are:")
    print(np.mean(scores_post), np.std(scores_post))
    
    pred_dur = model.predict(X_LSTM_dur)
    pred_post = model.predict(X_LSTM_post)
    
    
    figure = plt.figure(figsize=(7,5), dpi=300)
    plt.scatter(T_dur, Y_dur, s=1, c='b', alpha=0.8, label='Samples')
    plt.scatter(T_dur, pred_dur , s=1,c='tomato',alpha=0.8, label='Predictions')
    plt.legend()
    plt.grid()
    plt.ylim([-11,11])
    plt.xlabel("Days (2020-03-13 to 2022-01-27)")
    plt.ylabel("Normailzed Values")
    plt.title("Predictions & Samples of Pandemic Period")
    plt.savefig('fig5.png')
    plt.show()
    
    figure = plt.figure(figsize=(7,5), dpi=300)
    plt.scatter(T_post, Y_post, s=1, c='b', alpha=0.8, label='Samples')
    plt.scatter(T_post, pred_post , s=1,c='tomato',alpha=0.8, label='Predictions')
    plt.legend(loc='lower right')
    plt.grid()
    plt.ylim([-10,10])
    plt.xlabel("Days (2022-01-27 to 2023-12-25)")
    plt.ylabel("Normailzed Values")
    plt.title("Predictions & Samples of Pandemic Period")
    plt.savefig('fig6.png')
    plt.show()