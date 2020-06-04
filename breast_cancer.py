#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

#认识数据
#print("Cancer.keys: \n{}".format(cancer.keys()))
#print("Cancer.data.shape: \n{}".format(cancer.data.shape))
#print("Sample counts per class:\n{}".format({n:v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))#zip()函数用于将可迭代的对象作为参数，打包成一个元组，以列表的形式返回
#print("Cancer.feature_names: \n{}".format(cancer.feature_names))

#分类
from sklearn.model_selection import train_test_split
X_train, X_Test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)
from sklearn.neighbors import KNeighborsClassifier
train_accurancy = []
test_accurancy = []
neighbors_setting = range(1,11)
for neighbors in neighbors_setting:
    clf = KNeighborsClassifier(n_neighbors=neighbors)
    clf.fit(X_train, y_train)#拟合
    train_accurancy.append(clf.score(X_train,y_train))
    test_accurancy.append(clf.score(X_Test,y_test))
#print("Prediction:{}".format(clf.predict(X_Test)))
plt.plot(neighbors_setting, train_accurancy, label = "training accuracy")
plt.plot(neighbors_setting, test_accurancy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()