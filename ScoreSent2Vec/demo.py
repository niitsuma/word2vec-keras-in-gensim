#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""

"""

import logging
import sys
import os
import numpy 
from word2vec import Word2Vec, Sent2Vec, LineSentence, LineScoredSentence, ScoredSent2Vec

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))

sg_v=0
#sg_v=1

input_file = 'test.txt'
modelw = Word2Vec(LineSentence(input_file), size=100, window=5, sg=sg_v, min_count=5, workers=8)
modelw.save(input_file + '.model')
modelw.save_word2vec_format(input_file + '.vec')

sent_file = 'sent.txt'
models = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model',sg=sg_v)
models.save_sent2vec_format(sent_file + '.vec')


sents=list(LineSentence(sent_file))

def mysentscore1(sent):
    mywords=['the', 'and', 'of','with']
    return  [sent.count(w)/10.0 for w in mywords] ##better 1/10.0 to avoid overflow

sents_scores=[[s,mysentscore1(s)] for s in sents]
#print sents_scores
modelsc = ScoredSent2Vec(sents_scores, model_file=input_file+ '.model',sg=sg_v)
modelsc.save_sent2vec_format(sent_file + 'sc.vec')


N=7
mylabel=['A','B','B','A','A','B','B']
Y=mylabel
print len(Y)
X1 =[models.sents[n] for n in range(N)]
X2 =[modelsc.sents[n] for n in range(N)]


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score

scores= cross_val_score(LogisticRegression(), X1, Y, scoring='accuracy', cv=2)
print scores.mean()

scores= cross_val_score(LogisticRegression(), X2, Y, scoring='accuracy', cv=2)
print scores.mean()



program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)
