from numpy import *
import operator
from os import listdir
import numpy as np

# 对未知类别属性的数据集中的每个点依次执行以下操作：
# (1)计算已知类别数据集中的点与当前点之间的距离；
# (2)按照距离递增次序排序；
# (3)选取与当前点距离最小的走个点；
# (4)确定前灸个点所在类别的出现频率；
# (5)返回前k个点出现频率最高的类别作为当前点的预测分类。


# classifyO ()函数有4个输人参数:用于分类的输人向量是inX，输人的训练样本集为dataSet
# 标签向量为labels，最后的参数k表示用于选择最近邻居的数目，其中标签向量的元素数目和矩
# 阵dataSet的行数相同。使用欧氏距离公式计算两个向量点的距离
def classify0(inX, dataSet, labels, k):
    # 求矩阵的行数
    dataSetSize = dataSet.shape[0]
    # ile(inX, (dataSetSize,1))以inX的第一列元素作为源数据生成以[dataSetSize(生成矩阵的行数),column（生成矩阵的列的重复次数）]
    # 计算每个向量的横坐标和列坐标的差值
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    # 计算横轴坐标的各自的平方
    sqDiffMat = diffMat**2
    # 计算横轴坐标的平方和
    sqDistances = sqDiffMat.sum(axis=1)
    # 计算分类坐标到每个已知分类坐标的点的欧氏距离
    distances = sqDistances**0.5
    # 按照距离的从小到大排序，返回到每组的坐在的行数值  例如：
    # [4.1         4.12310563  5.38516481  5.34883165  5.34528764]
    # [0 1 4 3 2]
    sortedDistIndicies = distances.argsort()
    classCount={}
    # 比较前K的最近点中的各类的个数进行分类
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    # 对字典进行从大到小的排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]
def createDataSet():
    # 这四组数据，每组数据有两个我们已知的属性或者特征值，每组包含一个不同的数据
    # 我们可以将其想象为某个日志文件中不同的测试点或者入口
    group =np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1],[0,0.11]])
    # 包含上面的每组数据点的标签信息
    labels = ['A','A','B','B','B']
    return group, labels
# Test Code
group,labels= createDataSet()
print(classify0([0,0],group,labels,4))
# datingClassTest()