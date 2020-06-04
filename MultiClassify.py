#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import mglearn
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
X, y =make_blobs(random_state=42)
#print(X.shape)#查看数据有多少样本，每个样本有多少特征
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(["Class 0", "Class 1", "Class 2"])
#plt.show()

#训练LinearSVC分类器
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X,y)
print("Cofficient shape:",linear_svm.coef_.shape)#由于是多分类，因此权重是一个矩阵（即coef_属性的值）；而截距是一维数组
print("Intercept shape:",linear_svm.intercept_.shape)

#可视化分类器
#mglearn.discrete_scatter(X[:,0], X[:,1], y)
#line = np.linspace(-15,15)#创建-15和15之间的均匀分布的数据点，默认100个
#for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b','r','g']):
 #   plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
#plt.ylim(-10,15)
#plt.xlim(-10,8)
#plt.xlabel("Feature 0")
#plt.ylabel("Feature 1")
#plt.legend(['Class 0', 'Class 1', 'Class 2', 'line class 0', 'line class 1', 'line class 2'], loc=(1.01,0.3))
#plt.show()

#在二维空间展示预测结果
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=0.7)
mglearn.discrete_scatter(X[:,0], X[:,1], y)
line = np.linspace(-15,15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b','r','g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Class 0', 'Class 1', 'Class 2', 'line class 0', 'line class 1', 'line class 2'], loc=(1.01,0.3))
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()