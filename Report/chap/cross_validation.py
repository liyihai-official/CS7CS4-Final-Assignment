import pandas as pd
import numpy as np
import sys, math
import matplotlib.pyplot as plt
import tensorflow as tf
epoch = 10

plt.rc('font', size=18); plt.rcParams['figure.constrained_layout.use'] = True

import preprocessing
isolate_station = False
y, t, id, dt, station_info = preprocessing.main('cross_validation', isolate_station)

#plot extracted data
# plt.scatter(t[id==10], y[id==10], c='b', marker='+',s=2)
# plt.scatter(t[id==4], y[id==4], c='r', marker='+',s=2); plt.show()


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





from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
def ridge_model():
    def regression_model(C, input_shape, name_model):
        # 参数设置
        input_shape = [input_shape]  # 根据你的数据集输入特征的数量

        if name_model == "Lasso":
            regularizer = tf.keras.regularizers.l1(1/(2*C))
        else:
            regularizer = tf.keras.regularizers.l2(1/(2*C))

        # 创建一个简单的线性模型
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                units=1,  # 只有一个单元，表示是线性模型
                input_shape=input_shape,
                activation='linear',  # 线性激活函数
                kernel_regularizer=regularizer  # L1 L2 正则化
            )
        ])
        
        # 编译模型
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),  # 随机梯度下降优化器
                    loss='mean_squared_error',
                    metrics=['mse']
                    )  # 均方误差损失函数
        model.summary()
        return model
            
    def k_fold_cross_validation(X, y, C_vals, p, name_model):
        # Initializing the MSE and standard error
        mean_error = []
        std_error = []
        for C in C_vals:
            # Polynomial Featuring
            XX = PolynomialFeatures(p).fit_transform(X)
                    
            mean_square_error_temp = []
            kf = KFold(n_splits=5)
            # Training the model and applying teh cross validation
            for train, test in kf.split(XX):
                # Choosing a model
                # model = Lasso(alpha=1/(2*C), max_iter=10000) if name_model =="Lasso" else Ridge(alpha=1/(2*C))
                model = regression_model(C, XX.shape[1], name_model)
                
                # 训练模型
                model.fit(XX[train], y[train], 
                        epochs=epoch, 
                        batch_size=32, 
                        validation_split=0.2,
                        verbose=2
                        )  # epochs和batch_size根据需要调整
                
                predictions = model.predict(XX[test])
                mean_square_error_temp.append(mean_squared_error(y[test],predictions))
            mean_error.append(np.array(mean_square_error_temp).mean())
            std_error.append(np.array(mean_square_error_temp).std())

        return mean_error, std_error
    
    mmse_p1, mstde_p1 = k_fold_cross_validation(X, Y, C_vals = [1e-4, 1e-3, 1e-2, 1, 10], p=1, name_model="Lasso")
    mmse_p2, mstde_p2 = k_fold_cross_validation(X, Y, C_vals = [1e-4, 1e-3, 1e-2, 1, 10], p=2, name_model="Lasso")
    mmse_p1_r, mstde_p1_r = k_fold_cross_validation(X, Y, C_vals = [1e-4, 1e-3, 1e-2, 1, 10], p=1, name_model="Ridge")
    mmse_p2_r, mstde_p2_r = k_fold_cross_validation(X, Y, C_vals = [1e-4, 1e-3, 1e-2, 1, 10], p=2, name_model="Ridge")

    return mmse_p1, mstde_p1, mmse_p2, mstde_p2, mmse_p1_r, mstde_p1_r, mmse_p2_r, mstde_p2_r


import tensorflow.keras as keras
import tensorflow.keras.layers as layers
def lstm_model():
    
    def lstm_model(n_features, n_units):
        # 构建模型
        model = keras.Sequential()
        model.add(layers.LSTM(n_units, input_shape=(1, n_features)))
        model.add(layers.Dense(1))  # 因为您的目标输出是单个值

        model.compile(optimizer='adam', loss='mse', metrics=['mse']) # 选择合适的优化器和损失函数
        # model.summary()
        return model
            
    def k_fold_cross_validation(X, y, n_units):
        n_features = X.shape[2]
        
        # Initializing the MSE and standard error
        mean_error = []
        std_error = []
        for n_unit in n_units:
            mean_square_error_temp = []
            kf = KFold(n_splits=5)
            # Training the model and applying teh cross validation
            for train, test in kf.split(X):
                model = lstm_model(n_features, n_unit)
                
                # 训练模型
                model.fit(X[train], y[train], 
                        epochs=epoch, 
                        batch_size=32, 
                        validation_split=0.2,
                        verbose=2
                        )  # epochs和batch_size根据需要调整
                
                predictions = model.predict(X[test])
                mean_square_error_temp.append(mean_squared_error(y[test],predictions))
            mean_error.append(np.array(mean_square_error_temp).mean())
            std_error.append(np.array(mean_square_error_temp).std())

        return mean_error, std_error
    
    
    X_LSTM = X.reshape((X.shape[0], 1, X.shape[1]))
    mse_lstm, stde_lstm = k_fold_cross_validation(X_LSTM, Y, [1,10,50,100,1000])
    return mse_lstm, stde_lstm
    
def main():
    print("Start 5-fold cross validation to Regression Models")
    mmse_p1, mstde_p1, mmse_p2, mstde_p2, mmse_p1_r, mstde_p1_r, mmse_p2_r, mstde_p2_r = ridge_model()
    print("Start 5-fold cross validation to LSTM Models")
    mse_lstm, stde_lstm = lstm_model()
    print("SUCCESS")

if __name__ == '__main__':
    main()