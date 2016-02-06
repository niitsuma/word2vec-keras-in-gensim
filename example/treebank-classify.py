#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

import nltk
import gensim
from word2veckeras.doc2veckeras import SentenceClassifier,Doc2VecClassifier
from word2veckeras.treebank import TreeBank

treebank=TreeBank()
Xtest,Ytest=treebank.sents_labels('test',only_root=False,pos_neg_label=False)

X,Y        =treebank.sents_labels('train',only_root=False,pos_neg_label=False)

# X,Y        =treebank.sents_labels('dev',only_root=False,pos_neg_label=False)
# n_sample=300
# X=X[:n_sample]
# Y=Y[:n_sample]


clf1=SentenceClassifier( doc2vec=gensim.models.doc2vec.Doc2Vec() )
clf1.fit(X,Y)
print clf1.score(Xtest,Ytest)


from sklearn.grid_search import GridSearchCV,ParameterSampler, ParameterGrid

clf2=Doc2VecClassifier()
tuned_parameters = [{'dm':[1],'dm_concat':[0,1],'size': [200,300,400], 'window':[4,8],'min_count':[0,9],'sample':[0,1e-5],'iter':[1]}]
#tuned_parameters = [{'dm':[0,1],'size': [100,200]}]
clf2 = GridSearchCV(clf2, tuned_parameters,cv=3,n_jobs=4,verbose=1)
clf2.fit(X,Y)
print clf2.best_estimator_
print clf2.best_params_
print clf2.best_score_

print clf2.best_estimator_.fit(X,Y)
print clf2.best_estimator_.score(Xtest,Ytest)
