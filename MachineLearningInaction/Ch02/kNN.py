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


# 该函数的输人为文件名字符串，输出为训练样本矩阵和类标签向量
# 将文件中的内容转换为numpy矩阵，生成样本训练矩阵和样本所属的类的标签
def file2matrix(filename):
    fr = open(filename)
    # get the number of lines in the file
    numberOfLines = len(fr.readlines())
    # prepare matrix to return
    # 矩阵的另一维的维数设定为3，这是根据样本数据来决定的
    returnMat =np.zeros((numberOfLines,3))
    # prepare labels return
    classLabelVector = []
    fr = open(filename)
    index = 0
    data={}
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1] not in data):
            data[listFromLine[-1]]=len(data)+1
        classLabelVector.append(data[listFromLine[-1]])
        index+=1
    return returnMat,classLabelVector


# 数据的归一化处理
# newValue  =  {oldValue-min)/ (max-min)
# return  normDataSet,  ranges,  minVals
# 在函数autoNorm中，我们将每列的最小值放在变量minVals中，将最大值放在变量
# maxVals ,其中如dataSet.min(0)中的参数0使得函数可以从列中选取最小值，而不是选取当
# 前行的最小值。然后，函数计算可能的取值范围，并创建新的返回矩阵。正如前面给出的公式，
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    # element wise divide
    return normDataSet, ranges, minVals
   
def datingClassTest():
    hoRatio = 0.50
    # hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)
# 该函数创建1x1024的numpy数组，然后打开给定的文件，循环读出文件的前32行，
# 并将每行的头32个字符值存储在numpy数组中，最后返回数组
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    # load the training set
    m = len(trainingFileList)
    trainingMat =np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print( "\nthe total error rate is: %f" % (errorCount/float(mTest)))



from PIL import Image
from pylab import *
import numpy as np

#实际中的图片的测试
def ImageClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    # load the training set
    m = len(trainingFileList)
    trainingMat =np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    im = np.array(Image.open(r'C:\Users\陆建华\Pictures\IMG20170410154737(32pxX32px).jpg'))  # 打开图像，并转成灰度图像
    data = im[:,:, 0]
    classifierResult = classify0(data.reshape(1,1024), trainingMat, hwLabels, 3)
    print("**************")
    print(classifierResult)


ImageClassTest()