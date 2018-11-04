# 201834861dongxiao
DataMining Homwork
文件简介：
KNN-News文件夹，包括：数据集跟源代码
实验报告
运行结果截图

步骤一：
构建VSM,借助sklearn里面的TfidfVectorizer来实现。
步骤二：
借助划分好的数据集：
20news-bydate-test
20news-bydate-train
通过计算向量之间的距离度量相似性原理，设置k值，按相似度从大到小排序，看排在前k个样本属于哪个类型的最多，就把测试集样本分到那个类型。
