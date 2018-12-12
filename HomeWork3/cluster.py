# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:18:23 2018

@author: DX
"""

import numpy as np
from sklearn.cluster import KMeans,AffinityPropagation,MeanShift,SpectralClustering,DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer 
from nltk.tokenize import word_tokenize
from sklearn.metrics.cluster import normalized_mutual_info_score
import json
def K_means(text,labels,k):
    km=KMeans(n_clusters=k)
    result_label=km.fit_predict(text)
    print('K-means的准确率:', normalized_mutual_info_score(result_label, lables))
def A_ffinityPropagation(text,labels,k):
    ap = AffinityPropagation(damping=0.55, max_iter=575, convergence_iter=575, copy=True, preference=None,
                             affinity='euclidean', verbose=False)
    result_ap = ap.fit_predict(text)
    print('AffinityPropagation算法的准确率:', normalized_mutual_info_score(result_ap, labels))

def M_eanShift(text, labels, k):
    ms = MeanShift(bandwidth=0.65, bin_seeding=True)
    result_ms = ms.fit_predict(text)
    print('meanshift算法的准确率:', normalized_mutual_info_score(result_ms, labels))

def S_pectralClustering(text, labels, k):
    sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', n_neighbors=4, eigen_solver='arpack', n_jobs=1)
    result_sc = sc.fit_predict(text)
    print('SpectralClustering算法的准确率:', normalized_mutual_info_score(result_sc, labels))

def D_BSCAN(text, labels, k):
    db = DBSCAN(eps=0.7, min_samples=1)
    result_db = db.fit_predict(text)
    print('DBSCAN算法的准确率:', normalized_mutual_info_score(result_db, labels))

def A_gglomerativeClustering(text, labels, k):
    ac = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    result_ac = ac.fit_predict(text)
    print('AgglomerativeClustering算法的准确率:', normalized_mutual_info_score(result_ac, labels))

def G_aussianMixture(text, labels, k):
    gm = GaussianMixture(n_components=k, covariance_type='diag', max_iter=20, random_state=0)
    gm.fit(text)
    result_gm = gm.predict(text)
    print('GaussianMixture算法的准确率:', normalized_mutual_info_score(result_gm, labels))    
    
 
txt=[]
lables=[]    
text=[]
for line in open('Tweets.txt', 'r').readlines():
    dic=eval(line)
    txt.append(dic["text"])
    lables.append(dic["cluster"])
#分词器
tfidfvectorizer = TfidfVectorizer(tokenizer=word_tokenize, stop_words='english')
#转化成向量
text = tfidfvectorizer.fit_transform(txt).toarray()
vectorizer=CountVectorizer()
transformer=TfidfTransformer()
k=89
K_means(text,lables,k)
A_ffinityPropagation(text, lables, k)
M_eanShift(text, lables, k)
S_pectralClustering(text, lables, k)
D_BSCAN(text, lables, k)
A_gglomerativeClustering(text,lables,k)
G_aussianMixture(text,lables,k)