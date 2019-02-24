# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 19:48:08 2019

@author: Georgia Sarri, Leon Kalderon, George Vafeidis
"""


import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import scipy
import time
import nltk
import re

from bs4 import BeautifulSoup
import urllib3
from sklearn.feature_extraction.text import TfidfVectorizer
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

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score



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
#path = 'C:\\Users\\Georgia.Sarri\\Documents\\Msc\\5th\\TextAnalytics\\Assignmnets\\Untitled Folder\\ABSA16_Laptops_Train_SB1_v2.xml'
path = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\Assignment 2\\'
#path = r'C:\temp\\'

train_data = pd.DataFrame()

#for file, flag in [('ABSA16_Laptops_Train_SB1_v2.xml', 'C'), ('ABSA16_Restaurants_Train_SB1_v2.xml','R')]:
for file, flag in  [('ABSA16_Restaurants_Train_SB1_v2.xml','R')]:
    data = open(path+file, encoding="utf8").read()
    soup = BeautifulSoup(data, "lxml")
    sentences = soup.find_all("sentence")
    train_data = train_data.append(read_data(sentences, flag))
    
#==== LOAD VALIDATION DATA ====
### Concat with train data and some preprocessing
http = urllib3.PoolManager()
url = 'http://alt.qcri.org/semeval2016/task5/data/uploads/trial-data/english-trial/'
validation_data = pd.DataFrame()

for file, flag in [('restaurants_trial_english_sl.xml', 'R')]:
    response = http.request('GET', url+file)
    soup = BeautifulSoup(response.data, "lxml")
    sentences = soup.find_all("sentence")
    validation_data = validation_data.append(read_data(sentences, flag))
    
train_data = train_data.append(validation_data)
            
train_data.dropna(axis=0, how='any', inplace=True)
X_train = list(itertools.chain.from_iterable(train_data[['Text']].values.tolist()))
Y_train = list(itertools.chain.from_iterable(train_data[['Polarity']].values.tolist()))

train_data[train_data['Polarity']=='neutral']
train_data.groupby(['Polarity']).count()
train_data.columns


X_count = []
for idx,w in enumerate(X_train):
    X_count.extend( [1 if sum([str(c).isupper() for c in w])/len(w) > 0.5 else 0 ])
    w = remove_punctuation(nltk.wordpunct_tokenize(w.lower()))
    X_train[idx] = ' '.join(token for token in w)

X_train_copy = X_train.copy()

### LOAD TEST DATA ###
### And some preprocessing
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
        
#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
#X_test_tfidf = vectorizer.fit_transform(X_test)
X_test_copy = X_test.copy()

le = preprocessing.LabelEncoder()
le.fit(Y_test)
Y_test = le.transform(Y_test)

#X_test_category_dummies = pd.get_dummies(test_data['Category'])
#X_test_subcategory_dummies = pd.get_dummies(test_data['Subcategory'])
#X_test_hasCaps_dummies = pd.get_dummies(X_count_test)
#X_test_final = sparse.hstack((X_test_tfidf, np.array(X_test_category_dummies), np.array(X_test_subcategory_dummies)))

    
# Embeddings

file1 = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\glove.6B.300d.txt'
file2 = r'C:\MSc Data Science\Second Year 2nd Quarter\Text Analytics\Assignments\gensim_glove_vectors.txt'
#file1 = r'C:\temp\glove.6B\glove.6B.50d.txt'
#file2 = r'C:\temp\glove.6B\gensim_glove_vectors.txt'

glove2word2vec(glove_input_file=file1, word2vec_output_file=file2)
glove_model = KeyedVectors.load_word2vec_format(file2, binary=False)
#wv = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

### Train Data  ####
train_voc = []
for text in X_train_copy:
    words = text.split()
    for w in words:
        if w not in train_voc:
            train_voc.append(w)
train_voc_set = set(train_voc)
glove_voc_set = set(glove_model.vocab.keys())
print('\n\nTrain set vocabulary subset of glove vocabulary? {}'.format(train_voc_set.issubset(glove_voc_set)))
print('Size of training vocabulary {0:d}'.format(len(train_voc_set)))

text_embedings=list()
#i=0
for idx,text in enumerate(X_train_copy):
    text_in_words =text.split()
    # find only existing words in dictionary
    doc = [word for word in text_in_words if word in glove_model.vocab]
    #Return mean of embeddings
    text_vector = np.mean(glove_model[doc],  axis=0)
    #Create training vector in list form
    text_embedings.append(text_vector)

#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=8000 ,sublinear_tf=True)
#X_train_tfidf = vectorizer.fit_transform(X_train)
vectorizer.fit(X_train_copy)
print(X_train_copy[0])
print(len(vectorizer.vocabulary_))

## First (not most efficient) implementation of centroid taking into account TF-IDF
text_embeddings_tfidf = []
for idx,text in enumerate(X_train_copy):
    text_in_words =text.split()
    
    text_vector = np.zeros(glove_model.vector_size)
    denonimator = 0
    for word in text_in_words:
        if word in glove_model.vocab:#if not found in vocab ignore it
            if word in vectorizer.get_feature_names():
              word_idf = vectorizer.idf_[vectorizer.vocabulary_[word]]
              denonimator += word_idf
              text_vector = text_vector + word_idf * glove_model[word]
    # Calculate Centroids of each text and put it in the list    
    text_embeddings_tfidf.append(text_vector / denonimator)
    

le = preprocessing.LabelEncoder()
le.fit(Y_train)
Y_train = le.transform(Y_train)

### Test Data preparation  ####
train_voc = []
for text in X_test_copy:
    words = text.split()
    for w in words:
        if w not in train_voc:
            train_voc.append(w)
train_voc_set = set(train_voc)
glove_voc_set = set(glove_model.vocab.keys())
print('\n\nTrain set vocabulary subset of glove vocabulary? {}'.format(train_voc_set.issubset(glove_voc_set)))
print('Size of training vocabulary {0:d}'.format(len(train_voc_set)))

text_embedings_test=list()
for idx,text in enumerate(X_test_copy):
    text_in_words =text.split()
    # find only existing words in dictionary
    doc = [word for word in text_in_words if word in glove_model.vocab]
    #Return mean of embeddings that id tf centroid
    text_vector = np.mean(glove_model[doc],  axis=0)
    #Create training vector in list form
    text_embedings_test.append(text_vector)

#vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1200 ,sublinear_tf=True)
vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=8000 ,sublinear_tf=True)
#X_train_tfidf = vectorizer.fit_transform(X_train)
vectorizer.fit(X_test_copy)
print(X_test_copy[0])
print(len(vectorizer.vocabulary_))

## First (not most efficient) implementation of centroid taking into account TF-IDF
text_embeddings_test_tfidf = []
for idx,text in enumerate(X_test_copy):
    text_in_words =text.split()    
    text_vector = np.zeros(glove_model.vector_size)
    denonimator = 0
    for word in text_in_words:
        if word in glove_model.vocab:
            if word in vectorizer.get_feature_names(): #if not found in vocab ignore it
              word_idf = vectorizer.idf_[vectorizer.vocabulary_[word]]
              denonimator += word_idf
              text_vector = text_vector + word_idf * glove_model[word]
    # Calculate Centroids of each text and put it in the list    
    text_embeddings_test_tfidf.append(text_vector / denonimator)





# Choose accordingly which one you would like to train
# before running next section ofr training models
X_train_final = text_embedings.copy()
X_train_final = text_embeddings_tfidf.copy()
X_test_final = text_embedings_test.copy()
X_test_final = text_embeddings_test_tfidf.copy()


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
                    'C': np.linspace(0.1, 1, 5)
                    }
        },
        {
            'model_object': svm.SVC(),
            'hyper_params': {
                    'C': [1,1.4,1.6],
                    'kernel': ['linear'],
                    'gamma' : [0.01, 0.1, 1, 2]
                    }
        },
        {
            'model_object': RandomForestClassifier(random_state=0),
            'hyper_params': {
                    'max_depth':[100],#,200,500,None],
                    'max_features':[100,150],
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
        }                
    ]
    
# Random Search for best model
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
colors[7] = colors[8]
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
