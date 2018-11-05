from time import time
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter

# knn 实现
def kNN_Sparse(local_data_csr, query_data_csr, top_k):
    # 计算每个向量的均方和
    local_data_sq = local_data_csr.multiply(local_data_csr).sum(1)
    query_data_sq = query_data_csr.multiply(query_data_csr).sum(1)
     
    # 计算dot
    distance = query_data_csr.dot(local_data_csr.transpose()).todense()
     
    # 计算距离
    num_query, num_local = distance.shape
    distance = np.tile(query_data_sq, (1, num_local)) + np.tile(local_data_sq.T, (num_query, 1)) - 2 * distance
     
    # 获取前 k个索引
    topK_idx = np.argsort(distance)[:, 0:top_k]
    topK_similarity = np.zeros((num_query, top_k), np.float32)
    for i in range(num_query):
        topK_similarity[i] = distance[i, topK_idx[i]]
     
    return topK_similarity, topK_idx


def showmax(lt): 
    index1 = 0 #记录出现次数最多的元素下标
    max = 0 #记录最大的元素出现次数
    for i in range(len(lt)):
        flag = 0 #记录每一个元素出现的次数
        for j in range(i+1,len(lt)): #遍历i之后的元素下标
            if lt[j] == lt[i]:
                flag += 1 #每当发现与自己相同的元素，flag+1
        if flag > max: #如果此时元素出现的次数大于最大值，记录此时元素的下标
            max = flag
            index1 = i 
    return lt[index1] #返回出现最多的元素


def calTestRes(trnX,trnlabels,tstX, k):
    tstres=np.zeros(tstX.shape[0])    
#     计算topK索引和其相似性  
    topK_similarity, topK_idx = kNN_Sparse(trnX, tstX, k)     
    for i in range(tstX.shape[0]-1):
        tmp=topK_idx[i].tolist()
        tmp=tmp[0] 
        tstres[i] = showmax(trnlabels[tmp])        
    return tstres 

def checkRes(tstlab,tstres):
    nerr=0 
    for i in range(len(tstlab)):
        if (tstlab[i] != tstres[i]): nerr += 1.0    
    return ((len(tstlab)-nerr)/float(len(tstlab)))

print("加载训练数据 ...")
#t = time()
trndocs = load_files('./20news-bydate/20news-bydate-train')

tstdocs = load_files('./20news-bydate/20news-bydate-test')

#print("运行时间 {0}秒".format(time() - t))

max_features = 20000
print("向量化数据集 ...")
#t = time()
#定义参数 得到tf-idf的特征矩阵
vectorizer = TfidfVectorizer(max_df=0.4, 
                             min_df=2, #去除词频0.4-2之间的stopwords
                             max_features=max_features, #词频排序
                             encoding='latin-1')
ntrn=len(trndocs.data)
ntst=len(tstdocs.data)
#特征提取
data=trndocs.data+tstdocs.data
X = vectorizer.fit_transform(d for d in data)
trnX=X[:ntrn]
tstX=X[ntrn:]
#print("训练数据个数: %d, 训练数据特征数: %d" % trnX.shape)
#print("测试数据个数: %d, 测试数据特征数: %d" % tstX.shape)


#print("运行时间 {0}秒".format(time() - t))

trnlabels = trndocs.target
tstlabels = tstdocs.target
# knn
print("计算KNN ...")
#t = time()
for k in range(30,51):
    output=calTestRes(trnX,trnlabels,tstX, k)
    acc=checkRes(tstlabels,output)
    print('k = %d, 正确率: %f.'%(k,acc)) 
#print("运行时间 {0}秒".format(time() - t))  
