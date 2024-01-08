import pandas as pd
import numpy as np
import sys, math
import matplotlib.pyplot as plt

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True

import preprocessing

isolate_station = False
y, t, id, dt, station_info = preprocessing.main('pre', isolate_station)


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



X, Y, T, ID = feature_all_time_series(q = 3, lag = 3)

# 在训练之前，确保 x_train 和 x_test 的形状是 (样本数, 1, 特征数)
X_LSTM = X.reshape((X.shape[0], 1, 9))
X_LSTM.shape


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_LSTM, Y, test_size=0.2)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from tensorflow.keras.callbacks import EarlyStopping


# 模型参数
n_features = X_LSTM.shape[-1]  # 特征数量
n_units = 100     # LSTM层单元数量

# 构建模型
model = keras.Sequential()
model.add(layers.LSTM(n_units, input_shape=(1, n_features)))
model.add(layers.Dense(1))  # 因为您的目标输出是单个值

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse', metrics=['mse']) # 选择合适的优化器和损失函数
model.summary()
early_stopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0001, mode='min', verbose=1, restore_best_weights=True)


history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

model.save('model/LSTM')

fig = plt.figure(figsize=(5,5), dpi=100)
plt.plot(history.epoch, history.history['mse'], label = 'Training', color='r')
plt.plot(history.epoch, history.history['val_mse'], label = 'Validation', color='b')
plt.title('Training history')
plt.xlabel('Epoch')
plt.ylabel('Mean Square Error')
plt.legend(loc='upper right')
plt.grid()
plt.xlim([-5,70])
plt.savefig('fig4.png')
# plt.show()

model.evaluate(x_train, y_train);model.evaluate(x_test, y_test)