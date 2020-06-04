#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

#分析KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
fig, axes = plt.subplots(1,3,figsize=(15,4))
#创建1000个数据点，在-3和3之间均匀分布.(linspace函数)
line = np.linspace(-3,3,1000).reshape(-1,1)
for n_eighbors,ax in zip([1,3,9], axes):
    #利用1个、3个、9个邻居进行预测
    reg = KNeighborsRegressor(n_neighbors=n_eighbors)
    reg.fit(X_train, y_train)#拟合
    ax.plot(line,reg.predict(line))#预测
    ax.plot(X_train,y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test,y_test,'v',c=mglearn.cm2(1),markersize=8)
    ax.set_title("{} neighbors(s)\n train score:{:.2f} test score:{:.2f}".format(n_eighbors, reg.score(X_train,y_train),reg.score(X_test,y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target", "Teat data/target"], loc="best")
plt.show()
