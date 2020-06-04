#!/usr/bin/env python
#-*- coding = utf-8 -*-
import numpy as np
import pandas as pd
import mglearn
import matplotlib.pyplot as plt

#生成数据集
X1,y1 = mglearn.datasets.make_forge()
#print(format(X1.shape))
#绘图
mglearn.discrete_scatter(X1[:,0], X1[:,1], y1)#作图，参数与图像标记输出的位置有关
#plt.legend(["Class0", "Class1"], loc=4)#loc表示位置location
#plt.xlabel("First feature")
#plt.ylabel("Second feature")

#分类
#mglearn.plots.plot_knn_classification(n_neighbors=3)
#plt.show()
#k邻近回归
mglearn.plots.plot_knn_regression(n_neighbors=1)
plt.show()

#生成数据集
#X2,y2 = mglearn.datasets.make_wave(n_samples=40)
#print(format(X2.shape))
#plt.plot(X2, y2, 'o')
#plt.ylim(-3,3)
#plt.xlabel("Feature")
#plt.ylabel("Target")
#plt.show()