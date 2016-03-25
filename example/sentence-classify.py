#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

import nltk
import gensim
from word2veckeras.doc2veckeras import SentenceClassifier,Doc2VecClassifier

genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']
n_sample=40
sents_labels=  sum([list(zip(nltk.corpus.brown.sents(categories=[genres[i_g]])[:n_sample],[i_g]*n_sample)) for i_g in range(len(genres))],[])
X=[sl[0] for sl in sents_labels]
Y=[sl[1] for sl in sents_labels]

clf1=SentenceClassifier( doc2vec=gensim.models.doc2vec.Doc2Vec() )
clf1.fit(X,Y)
print clf1.score(X,Y)

from sklearn.grid_search import GridSearchCV,ParameterSampler, ParameterGrid

clf2=Doc2VecClassifier()
tuned_parameters = [{'dm':[1],'dm_concat':[0,1],'size': [200,300,400], 'window':[4,8],'min_count':[0,9],'sample':[0,1e-5],'iter':[1]}]
#tuned_parameters = [{'dm':[1],'size': [200,300,400]}]
clf2 = GridSearchCV(clf2, tuned_parameters,cv=3,n_jobs=-1,verbose=1)
clf2.fit(X,Y)
print clf2.best_estimator_
print clf2.best_params_
print clf2.best_score_
print clf2.score(X,Y)
