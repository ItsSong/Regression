#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#导入类并实例化
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
#对分类器进行拟合
clf.fit(X_train, y_train)
#进行预测
print("Test set prediction:{}".format(clf.predict(X_test)))
#评估精度
print("Test set accurancy:{:.2f}".format(clf.score(X_test, y_test)))

#作图
fig, axes = plt.subplots(1,3,figsize=(10,3))#显示图1行3列；每个图长10宽3
for n_neighbors, ax in zip([1,3,9], axes):
#拟合、实例化
 clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
 mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
 mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
 ax.set_title("{} neighbor(s)".format(n_neighbors))
 ax.set_xlabel("feature 0")
 ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()