# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:25:29 2018

@author: DX
"""

from numpy import *
from os import listdir,mkdir,path
import re
from nltk.corpus import stopwords
import nltk
import operator
import io

#文本数据预处理
#通过输入不同的参数20news-bydate-test、20news-bydate-train、20news-18828
#分别生成测试、训练、以及为建立词典做准备的预处理文件即processed_test、processed_train、processed
def processing():
    srcFilesList = listdir('20news-18828')#20news-bydate-train、20news-18828
    #print(len(srcFilesList))
    for i in range(len(srcFilesList)):
        #print(i)
        dataFilesDir = '20news-18828/' + srcFilesList[i] # 20个文件夹每个的路径
        dataFilesList = listdir(dataFilesDir)
        targetDir = 'processed/' + srcFilesList[i] # 20个新文件夹每个的路径
        if path.exists(targetDir)==False:
            mkdir(targetDir)
        else:
            print( '%s exists' % targetDir)
        for j in range(len(dataFilesList)):
            srcFile = '20news-18828/' + srcFilesList[i] + '/' + dataFilesList[j]
            targetFile= 'processed/' + srcFilesList[i]\
                + '/' + dataFilesList[j]
            fw = open(targetFile,'w')
            dataList =io.open(srcFile,'r',encoding='latin-1').readlines()
            for line in dataList:
                stopwords = nltk.corpus.stopwords.words('english') #去停用词
                porter = nltk.PorterStemmer()  #词干分析
                splitter = re.compile('[^a-zA-Z]')  #去除非字母字符，形成分隔
                words = [porter.stem(word.lower()) for word in splitter.split(line)\
                     if len(word)>0 and\
                     word.lower() not in stopwords]
                resLine = words 
                for word in resLine:
                    fw.write('%s\n' % word) #一行一个单词
            fw.close()
            print ('%s %s' % (srcFilesList[i],dataFilesList[j]))
#processing()
#创建词典createWordMap.txt
def createWordMap():
    countLine=0
    fr = open('D:\\DataMining\\HomeWork2\\createWordMap.txt','w')
    wordMap = {}
    newWordMap = {}
    fileDir = 'processed'
    sampleFilesList = listdir(fileDir)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = fileDir + '/' + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        for j in range(len(sampleList)):
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir).readlines():
                word = line.strip('\n')
                wordMap[word] = wordMap.get(word,0.0) + 1.0
    #过滤词频小于4的单词
    for key, value in wordMap.items():
        if value > 4:
            newWordMap[key] = value
    sortedNewWordMap = sorted(newWordMap.items())
    #print ('词典大小 : %d' % len(wordMap))
    #print ('新词典大小 : %d' % len(sortedNewWordMap))
    sortedWordMap = sortedNewWordMap
    for item in sortedWordMap:
        fr.write('%s %.1f\n' % (item[0],item[1]))
        countLine += 1
    #print( 'sortedWordMap size : %d' % countLine)
    return sortedNewWordMap
#createWordMap()
#对训练和测试预处理文件按照词典选取token
#参数fileDir：processed_test、processed_train
#分别得到new_processed_test、new_processed_train
def tokenWords():
    fileDir = 'processed_train'
    wordMapDict = {}
    sortedWordMap = createWordMap()
    for i in range(len(sortedWordMap)):
        wordMapDict[sortedWordMap[i][0]]=sortedWordMap[i][0]    
    sampleDir = listdir(fileDir)
    for i in range(len(sampleDir)):
        targetDir = 'new_processed_train' + '/' + sampleDir[i]
        srcDir = 'processed_train' + '/' + sampleDir[i]
        if path.exists(targetDir) == False:
            mkdir(targetDir)
        sample = listdir(srcDir)
        for j in range(len(sample)):
            targetSampleFile = targetDir + '/' + sample[j]
            fr=open(targetSampleFile,'w')
            srcSampleFile = srcDir + '/' + sample[j]
            for line in open(srcSampleFile).readlines():
                word = line.strip('\n')
                if word in wordMapDict.keys():
                    fr.write('%s\n' % word)
            fr.close()
#tokenWords()
#创建测试文件的分类标注文件AnnotationFile.txt
#标注：序号 所属类
def createFile():
    AnnotationFile = 'AnnotationFile' + '.txt'
    fr = open(AnnotationFile,'w')
    fileDir = 'new_processed_test'
    sampleFilesList=listdir(fileDir)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = fileDir + '/' + sampleFilesList[i]
        sampleList = listdir(sampleFilesDir)
        m = len(sampleList)
        for j in range(m):
            fr.write('%s %s\n' % (sampleList[j],sampleFilesList[i]))       
    fr.close()
        

#createFile()

# 统计训练样本中，每个目录下每个单词的出现次数, 及每个目录下的单词总数
def countWords(strDir):
    conWordsNum = {}#<类，单词总数>
    conWordsProb = {}#<类_单词 ,某单词出现次数>
    conDir = listdir(strDir)
    for i in range(len(conDir)):
        count = 0 # 记录每个类下单词总数
        sampleDir = strDir + '/' + conDir[i]
        sample = listdir(sampleDir)
        for j in range(len(sample)):
            sampleFile = sampleDir + '/' + sample[j]
            words = open(sampleFile).readlines()
            for line in words:
                count = count + 1
                word = line.strip('\n')                
                keyName = conDir[i] + '_' + word
                conWordsProb[keyName] = conWordsProb.get(keyName,0)+1 # 记录即每个类下每个单词的出现次数
        conWordsNum[conDir[i]] = count
        #print ('目录 %d 包含 %d' % (i,conWordsNum[conDir[i]]))
    #print ('目录下单词的大小: %d' % len(conWordsProb))
    return conWordsProb, conWordsNum
#countWords('new_processed_train')

#用贝叶斯对测试文档分类得到分类结果文件ResultFileNew
def Bayes(traindir,testdir,ResultFileNew):
    crWriter = open(ResultFileNew,'w')
   
    #返回词的出现次数,总词数
    conWordsProb, conWordsNum = countWords(traindir)

    #训练集的总词数
    trainTotalNum=0
    for i in conWordsNum.values():
        trainTotalNum=i+trainTotalNum
    #trainTotalNum = sum(cateWordsNum.values())
    #print('trainTotalNum: %d' % trainTotalNum)
    print("trainTotalNum:")
    print(trainTotalNum)

    #开始对测试样例做分类
    testDirFiles = listdir(testdir)
    for i in range(len(testDirFiles)):
        testSampleDir = testdir + '/' + testDirFiles[i]
        testSample = listdir(testSampleDir)
        for j in range(len(testSample)):
            testFilesWords = []
            sampleDir = testSampleDir + '/' + testSample[j]
            lines = open(sampleDir).readlines()
            for line in lines:
                word = line.strip('\n')
                testFilesWords.append(word)
            a = 0.0
            trainDirFiles = listdir(traindir)
            for k in range(len(trainDirFiles)):
                p = compute(trainDirFiles[k], testFilesWords,\
                                    conWordsNum, trainTotalNum, conWordsProb)
                if k==0:
                    a = p
                    bestCate = trainDirFiles[k]
                    continue
                if p > a:
                    a = p
                    bestCate = trainDirFiles[k]
            crWriter.write('%s %s\n' % (testSample[j],bestCate))
    crWriter.close()
#分别计算条件概率、先验概率
def compute(traindir,testFilesWords,conWordsNum,\
                    totalWordsNum,conWordsProb):
    prob = 0
    wordNumInCate = conWordsNum[traindir] # 类下单词总数 <类目，单词总数>
    for i in range(len(testFilesWords)):
        keyName = traindir + '_' + testFilesWords[i]
        if keyName in conWordsProb:
            testFileWordNumInCate = conWordsProb[keyName] # 类下词出现的次数
        else: testFileWordNumInCate = 0.0
        xcProb = log((testFileWordNumInCate + 0.0001) /(wordNumInCate + totalWordsNum))   
        prob = prob + xcProb
    res = prob + log(wordNumInCate) - log(totalWordsNum)
    return res
#计算准确率
def computeAccuracy(rightCate,resultCate):
    rightCateDict = {}
    resultCateDict = {}
    rightCount = 0.0

    for line in open(rightCate).readlines():
        (sampleFile,cate) = line.strip('\n').split(' ')
        rightCateDict[sampleFile] = cate
        
    for line in open(resultCate).readlines():
        (sampleFile,cate) = line.strip('\n').split(' ')
        resultCateDict[sampleFile] = cate
        
    for sampleFile in rightCateDict.keys():
        if (rightCateDict[sampleFile]==resultCateDict[sampleFile]):
            rightCount += 1.0
    accuracy = rightCount/len(rightCateDict)
    print ('accuracy: %f' % (accuracy))
    return accuracy
ResultFileNew = 'ResultFileNew' + '.txt'
Bayes('new_processed_train','new_processed_test',ResultFileNew)
rightCate = 'AnnotationFile'+'.txt'
resultCate = 'ResultFileNew'+'.txt'
computeAccuracy(rightCate,resultCate)