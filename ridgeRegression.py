#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

#岭回归波士顿房价
from sklearn.model_selection import train_test_split
X, y=mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

#线性回归
from sklearn.linear_model import LinearRegression
lr =LinearRegression().fit(X_train,y_train)

#岭回归
from sklearn.linear_model import Ridge #引入岭回归的包
ridge = Ridge().fit(X_train,y_train)
print("Training set score:{:.2f}".format(ridge.score(X_train,y_train)))#训练的精度
print("Test set score:{:.2f}".format(ridge.score(X_test,y_test)))#测试的精度

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")

ridge10 = Ridge(alpha=10).fit(X_train,y_train)
plt.plot(ridge10.coef_, '^',label="Ridge alpha=10")

ridge01 = Ridge(alpha=0.1).fit(X_train,y_train)
plt.plot(ridge01.coef_, 'v',label="Ridge alpha=0.1")

plt.plot(lr.coef_, 'o',label = "LinearRegression")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0,0,len(lr.coef_))
plt.ylim(-25,25)
plt.legend()
plt.show()
