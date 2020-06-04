#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

#引入logistic回归和线性支持向量机（二者都是分类算法）
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X,y = mglearn.datasets.make_forge()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

fig, axes = plt.subplots(1,2,figsize=(10,3))#定义显示图像的时候：1行2列，每个图长10宽3

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X_train,y_train)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=0.7)
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)#输出图像标记的参数
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend()

#使用LinearSVC分类
mglearn.plots.plot_linear_svc_regularization()
plt.show()

