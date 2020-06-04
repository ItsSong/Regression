#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

#线性回归，学习参数w和b(即斜率和截距）

#法1：直接调用包里的线性回归函数（可以显示图）
# mglearn.plots.plot_linear_regression_wave()
#plt.show()

#法2：（不能显示图，需要重新定义作图的命令）
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X,y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42)
lr = LinearRegression().fit(X_train, y_train)
print("lr.coef_:{}".format(lr.coef_))#线性回归的斜率被存储在coef_的属性内
print("lr.intercept_:{}".format(lr.intercept_))#截距被存储在intercept_属性中

#评估训练集和测试集的性能
print("Training accuracy:{:.2f}".format(lr.score(X_train,y_train)))
print("Test accuracy:{:.2f}".format(lr.score(X_test,y_test)))