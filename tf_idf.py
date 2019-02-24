# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:15:19 2019

@author: Georgia Sarri, Leon Kalderon, George Vafeidis
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import scipy
import nltk
import re
from bs4 import BeautifulSoup
import urllib3
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import itertools
from sklearn import preprocessing
from scipy import sparse
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, confusion_matrix
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from typing import List, Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.decomposition import TruncatedSVD

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
   
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=False)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="b")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="b",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="orange",
             label="Cross-validation score")

    plt.legend(loc="lower right")
    return plt

def read_data(sentences, flag):
    data = pd.DataFrame(columns=['sentence_id', 'Category', 'Subcategory', 'Target', 'Text', 'Polarity'])
    for sentence in sentences:
        sent_id = sentence.get('id')
        text = sentence.find('text').text
        opinions = sentence.find_all('opinion')
        if(len(opinions)==0):
            data = data.append({'sentence_id': sent_id, 
                                'Category': np.nan, 
                                'Subcategory': np.nan,
                                'Target': np.nan,
                                'Text': text, 
                                'Polarity': np.nan}, ignore_index=True)
        else:
            for opinion in opinions:
                category = opinion.get('category').split('#')[0]
                subcategory = opinion.get('category').split('#')[1]
                polarity = opinion.get('polarity')
                target = 'NA' if flag == 'C' else opinion.get('target')
                data = data.append({'sentence_id': sent_id, 
                                    'Category': category, 
                                    'Subcategory': subcategory, 
                                    'Target': target,
                                    'Text': text, 
                                    'Polarity': polarity}, ignore_index=True)
    return data


def remove_punctuation(words: List[str]) -> List[str]:
    """
    Remove punctuation from list of tokenized words

    :param List[str] words: A list of words(tokens)
    :return List[str] new_words: the list of words with all the punctuations removed.
    """
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

#==== LOAD TRAIN ====

#path = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign2\\'
path = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 2\\'

train_data = pd.DataFrame()

#for file, flag in [('ABSA16_Laptops_Train_SB1_v2.xml', 'C'), ('ABSA16_Restaurants_Train_SB1_v2.xml','R')]:
for file, flag in  [('ABSA16_Restaurants_Train_SB1_v2.xml','R')]:
 
    data = open(path+file, encoding="utf8").read()
    soup = BeautifulSoup(data, "lxml")
    sentences = soup.find_all("sentence")
    train_data = train_data.append(read_data(sentences, flag))

    
#==== LOAD VALIDATION DATA ====
http = urllib3.PoolManager()
url = 'http://alt.qcri.org/semeval2016/task5/data/uploads/trial-data/english-trial/'
validation_data = pd.DataFrame()

for file, flag in [('restaurants_trial_english_sl.xml', 'R')]:
    response = http.request('GET', url+file)
    soup = BeautifulSoup(response.data, "lxml")
    sentences = soup.find_all("sentence")
    validation_data = validation_data.append(read_data(sentences, flag))

train_data.dropna(axis=0, how='any', inplace=True)
X_train = list(itertools.chain.from_iterable(train_data[['Text']].values.tolist()))
Y_train = list(itertools.chain.from_iterable(train_data[['Polarity']].values.tolist()))

X_count = []
for idx,w in enumerate(X_train):
    X_count.extend( [1 if sum([str(c).isupper() for c in w])/len(w) > 0.5 else 0 ])
    w = remove_punctuation(nltk.wordpunct_tokenize(w.lower()))
    X_train[idx] = ' '.join(token for token in w) 
        
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)

idx = np.where(np.array(Y_train) == 'neutral')


x_train_neutral = X_train_tfidf[idx]
X_category_neutral = []#list(train_data['Category'].iloc[idx])
X_subcategory_neutral = []#list(train_data['Subcategory'].iloc[idx])
for i in range(8):
    X_train_tmp = x_train_neutral.copy()
    for index in range(len(X_train_tmp.data)):
        tmp = np.random.normal(0.0,0.05)
        X_train_tmp.data[index] += tmp if tmp > 0 else 0
    X_train_tfidf = sparse.vstack([X_train_tfidf, X_train_tmp])
    #X_train_tfidf = sparse.vstack([X_train_tfidf, x_train_neutral])
    X_category_neutral.extend(train_data['Category'].iloc[idx])
    X_subcategory_neutral.extend(train_data['Subcategory'].iloc[idx])
    
Y_train.extend(['neutral']*8*(sum(train_data['Polarity']=='neutral')))

le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)

X_category_dummies = pd.get_dummies(np.append(train_data['Category'], np.array(X_category_neutral)))
X_subcategory_dummies = pd.get_dummies(np.append(train_data['Subcategory'], np.array(X_subcategory_neutral)))
X_hasCaps_dummies = pd.get_dummies(X_count)
# X_subcategory_dummies = pd.get_dummies(train_data['Target'])
#X_train_final = sparse.hstack((X_train_tfidf, np.array(X_category_dummies), np.array(X_subcategory_dummies), np.array(X_hasCaps_dummies)))
X_train_final = sparse.hstack((X_train_tfidf, np.array(X_category_dummies), np.array(X_subcategory_dummies)))
#X_train_final = X_train_tfidf.copy()

print('Train Data Shape: {}'.format(X_train_final.shape))

### LOAD TEST DATA ###
#path_test = r'C:\Users\User\Desktop\MSc Courses\Natural Language Processing\assign2\EN_REST_SB1_TEST.xml'
path_test = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 2\EN_REST_SB1_TEST.xml'

data = open(path_test, encoding="utf8").read()
soup = BeautifulSoup(data, "lxml")
sentences = soup.find_all("sentence")
test_data = read_data(sentences, flag)

test_data.dropna(axis=0, how='any', inplace=True)

X_test = list(itertools.chain.from_iterable(test_data[['Text']].values.tolist()))
Y_test = list(itertools.chain.from_iterable(test_data[['Polarity']].values.tolist()))

X_count_test = []
for idx,w in enumerate(X_test):
    X_count_test.extend( [1 if sum([str(c).isupper() for c in w])/len(w) > 0.5 else 0 ])
    w = remove_punctuation(nltk.wordpunct_tokenize(w.lower()))
    X_test[idx] = ' '.join(token for token in w) 
        
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
X_test_tfidf = vectorizer.fit_transform(X_test)


le = preprocessing.LabelEncoder()
le.fit(Y_test)
Y_test = le.transform(Y_test)

X_test_category_dummies = pd.get_dummies(test_data['Category'])
X_test_subcategory_dummies = pd.get_dummies(test_data['Subcategory'])
X_test_hasCaps_dummies = pd.get_dummies(X_count_test)
X_test_final = sparse.hstack((X_test_tfidf, np.array(X_test_category_dummies), np.array(X_test_subcategory_dummies)))

# SVD Approach mainly for SVM
#svd = TruncatedSVD(n_components=350, random_state=1)
#X_train_svd = svd.fit_transform(X_train_final)
#X_test_svd = svd.transform(X_test_final)


#===== MODELS =======
def get_number_of_iterations(hyper_param_dict):
    n_iter = 1
    for param, values in hyper_param_dict.items():
        n_iter *= len(values)
    return n_iter

def hypertuning_rscv(est, p_distr, X, y):
    nbr_iter = get_number_of_iterations(p_distr)
    #CV = Cross-Validation ( here using Stratified KFold CV)
    cv = StratifiedKFold(n_splits=5, shuffle = True , random_state=42)
    
    rdmsearch = RandomizedSearchCV(est, param_distributions=p_distr,
                                  n_jobs=1, n_iter=nbr_iter, cv=cv, scoring='f1_micro', random_state=1)
    
    start = time.time()
    rdmsearch.fit(X,y)
    print('hyper-tuning time : %d seconds' % (time.time()-start))
    start = 0
    ht_params = rdmsearch.best_params_
    ht_score = rdmsearch.best_score_
    
    return ht_params, ht_score

models = [        
        {
            'model_object': LogisticRegression(random_state=0),
            'hyper_params': {
                    'dual': [True,False],
                    'max_iter': [100,110,120,130,140],
                    'C': np.linspace(0.1, 1, 10)
                    }
        },
        {
            'model_object': MultinomialNB(),
            'hyper_params': {
                    'alpha':np.linspace(0.1,2,20)
                        }
                    
        },
        {
            'model_object': svm.SVC(),
            'hyper_params': {
                    'C': [1,1.2,1.5],
                    'kernel': ['linear'],
                    'gamma' : [0.01, 0.1, 1, 2]
                    }
        },
        {
            'model_object': RandomForestClassifier(random_state=0),
            'hyper_params': {
                    'max_depth':[100],#,200,500,None],
                    'max_features':[500],
#                    'max_features':[100, 500, 1000],
#                    'criterion':['gini','entropy'],
                    'criterion':['gini'],
#                    'n_estimators':[500, 1000]
                    'n_estimators':[1000]
                    #'min_samples_leaf':randint(1,10)
                            }
        },
        {   
            'model_object': BaggingClassifier(base_estimator= DecisionTreeClassifier(), random_state=1),
            'hyper_params': {
                    'n_estimators': [1, 10, 50, 60, 5, 20],
                    
                    }
        },        
        {
            'model_object': ExtraTreesClassifier(), #{'n_estimators': 50, 'max_features': 100} f1~87
            'hyper_params': {
                   'n_estimators':[1000],
                   'max_features':[100]
#                   'n_estimators':[50, 200, 500, 1000],
#                   'max_features':[10, 100, 500]
                    }            
        }
        
    ]
    
for model in models:
  model_name = model['model_object'].__class__.__name__
  print(model_name)
  rf_parameters, rf_ht_score = hypertuning_rscv(est=model['model_object'], p_distr=model['hyper_params'], X=X_train_final, y=Y_train)  
  model['best_params'] = rf_parameters
  print(rf_parameters)
  print('Hyper-tuned model f1 average micro score :')
  print(rf_ht_score)


## PRECSION RECALL
# Plot Precision-Recall curves
X_train = X_train_final
y_train = Y_train
np.seterr(all='ignore')

y_test_encoded = pd.get_dummies(Y_test)

roc = plt.axes()
lc = plt.figure()
colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))
#colors[4] = colors[8]
j = 0
best_trained_models=list()
for model in models:
    model_name = model['model_object'].__class__.__name__
    model_param = model['best_params']
    if model_name=='SVC':
      model_param.update({'probability':True})
    est = model['model_object'].set_params(**model_param)
    
    est.fit(X_train,y_train)
    pred = est.predict_proba(X_test_final)
    best_trained_models.append(est)
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_test_encoded[i],
                                                            pred[:,i])
        average_precision[i] = average_precision_score(y_test_encoded[i], pred[:,i])
    
    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], threshold = precision_recall_curve(y_test_encoded.values.ravel(),
        pred.ravel())
    average_precision["micro"] = average_precision_score(y_test_encoded, pred,
                                                         average="micro")
    
    roc.step(recall['micro'], precision['micro'], color=colors[j], alpha=0.2,
             where='post', label=model_name)
    roc.fill_between(recall["micro"], precision["micro"], alpha=0.0002, color=colors[i])
    j+=1

    title = "Learning Curves {}".format(model_name)
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    plot_learning_curve(est, title, X_train, y_train, ylim=(0.7, 1.01), cv=cv)
    
roc.set_xlabel('Recall')
roc.set_ylabel('Precision')
roc.set_ylim([0.0, 1.05])
roc.set_xlim([0.0, 1.0])
roc.set_title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))

roc.legend(loc='lower right')    
plt.show()

for model in best_trained_models:
    confusion_matrix1 = confusion_matrix(Y_test,model.predict(X_test_final))
    print(model)
    print(confusion_matrix1)
