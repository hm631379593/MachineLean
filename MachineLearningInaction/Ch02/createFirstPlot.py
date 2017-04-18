
from numpy import *
import kNN
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
# 通过单色简单的显示散点图
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# 通过后面的标签用不同的颜色来表示不同的分类的类别
ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
ax.axis([-2,25,-0.2,2.0])
plt.xlabel('Percentage of Time Spent Playing Video Games')
plt.ylabel('Liters of Ice Cream Consumed Per Week')
plt.show()
