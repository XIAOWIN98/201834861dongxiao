# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 08:25:57 2018

@author: DX
"""
#TEST TEXTBLOG
from textblob import TextBlob
import os
path =r'D:\DataMining\20news-18828'
#zen = TextBlob("Beautiful is better than ugly. "
 #              "Explicit is better than implicit. "
  #              "Simple is better than complex.")
#print(zen.words)
#print(zen.tags)

def getFileNum(path):
	##BEGIN
	##统计path路径中文件的个数并返回
	allFileNum = 0
	for dirpath,dirnames,filenames in os.walk(path):
		#print(dirpath)
		#print(dirnames,filenames)
		for filename in filenames:
			allFileNum=allFileNum+1
			#print(os.path.join(dirpath,filename))           
	#files = os.listdir(path)#返回path指定的文件夹包含的文件或文件夹的名字的列表。
	#for file in files:
		#print(file)
	#for f in files:
		#if(os.path.isfile(path+'/'+f)):
			#allFileNum = allFileNum + 1
	return allFileNum
	##END
    
print(getFileNum(path))
