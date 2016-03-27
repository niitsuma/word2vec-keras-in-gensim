import sys
import numpy as np
import gensim
from word2veckeras.doc2veckeras import Doc2VecKeras

def compare_d2v(d2v1,d2v2):
    return sum([np.linalg.norm(d2v1.docvecs[n]-d2v2.docvecs[n])  for n in range(len(d2v1.docvecs)) ])/len(d2v1.docvecs)

input_file = 'test.txt'
doc1=gensim.models.doc2vec.TaggedLineDocument(input_file)

parameters = [{'size':[5],'dm':[0,1],'dm_concat':[0,1],'hs':[0,1],'negative':[0,5] }]
from sklearn.grid_search import ParameterGrid
for param in ParameterGrid(parameters):
    if (param['hs']==0 and param['negative']==0) or (param['dm']==0 and param['dm_concat']==0) :
        continue
    
    print param
    dvk=Doc2VecKeras(doc1,**param)
    dv =gensim.models.doc2vec.Doc2Vec(doc1,**param)
    print compare_d2v(dv,dvk)


