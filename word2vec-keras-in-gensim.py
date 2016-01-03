#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

from numpy import zeros, random, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide

import gensim.models.word2vec 

from six.moves import xrange, zip
from six import string_types, integer_types, itervalues

import sys
import random

import numpy as np


import keras.constraints

from keras.utils.np_utils import accuracy
from keras.models import Graph,Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten 
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse


def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None):

    if word not in model.vocab:
        return
    predict_word = model.vocab[word]  # target word (NN output)
    
    if model.hs:
        y=np.zeros((len(model.vocab)), dtype=REAL)
        x1=np.zeros((len(model.vocab)), dtype=REAL)
        for k,i in enumerate(predict_word.code):
            y[predict_word.point[k]]=i
            x1[predict_word.point[k]]=1
        # x1[predict_word.point]=1
        # y[predict_word.point]=predict_word.code
        x0=context_index
        #x1=predict_word.index
        return x0,x1,y
        #return (np.array([[x0]]),np.array([x1]),np.array([y]))

    # if model.negative:



def train_batch_sg(model, sentences, alpha, work=None,batch_size=100):
    
    batch_count=0

    # train_x0=[]
    # train_x1=[]
    # train_y=[]

    idxs = range(batch_size)
    #print idxs
    random.shuffle(idxs)
    #print idxs
    train_x0=[[]]*batch_count
    train_x1=[[]]*batch_count
    train_y=[[]]*batch_count
    
    while 1:

        for sentence in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        xy=train_sg_pair(model, model.index2word[word.index], word2.index, alpha)
                        if xy !=None:
                            (x0,x1,y)=xy

                            if len(train_x0)< batch_size:
                                train_x0.append([x0])
                                train_x1.append(x1)
                                train_y.append(y)
                            else:
                                #print idxs[batch_count]
                                train_x0[idxs[batch_count]]=[x0]
                                train_x1[idxs[batch_count]]=x1
                                train_y[idxs[batch_count]]=y
                                
                            batch_count += 1
                            
                            if batch_count >= batch_size :
                                #yield { 'index':np.array(x0), 'point':np.array(x1), 'code':np.array(y)}
                                #print len(train_x0)
                                # print train_x0[idxs]
                                # sys.exit()
                                #print train_x0
                                

                                # idxs = range(len(train_x0))
                                # random.shuffle(idxs)
                                # train_x0=train_x0[idxs]
                                # train_x1=train_x1[idxs]
                                # train_y=train_y[idxs]
                                
                                #print len(train_x0)
                                #yield { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y)}
                                
                                ax0=np.array(train_x0)
                                ax1=np.array(train_x1)
                                ay=np.array(train_y)
                                yield { 'index':ax0, 'point':ax1, 'code':ay}
 
                                
                                batch_count=0
                                random.shuffle(idxs)
                                # train_x0=[]
                                # train_x1=[]
                                # train_y=[]
                            
                            #yield { 'context_index':x0, 'predict_word.point':x1, 'predict_word.code':y}
                            #yield { 'index':x0, 'point':x1, 'code':y}

class Word2VecKeras(gensim.models.word2vec.Word2Vec):

    def build_keras_model(self):

        vocab_size=len(self.vocab)
        code_dim=vocab_size

        self.kerasmodel = Graph()
        self.kerasmodel.add_input(name='point', input_shape=(code_dim,), dtype=REAL)
        self.kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
        self.kerasmodel.add_node(self.kerasmodel.inputs['point'],name='pointnode')
        self.kerasmodel.add_node(Embedding(vocab_size, self.vector_size, input_length=1),name='embedding', input='index')
        self.kerasmodel.add_node(Flatten(),name='embedflatten',input='embedding')
        self.kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0)), name='sigmoid', input='embedflatten')
        self.kerasmodel.add_output(name='code',inputs=['sigmoid','pointnode'], merge_mode='mul')
        self.kerasmodel.compile('rmsprop', {'code':'mse'})

    def train(self, sentences, total_words=None, word_count=0, chunksize=100, total_examples=None, queue_factor=2, report_delay=1):
        #print 'Word2VecKerastrain'
        batch_size=500
        if self.sg:
            self.build_keras_model()
            samples_per_epoch=int(self.window*2*sum(map(len,sentences))/batch_size)
            print 'samples_per_epoch',samples_per_epoch
            self.kerasmodel.fit_generator(train_batch_sg(self, sentences, self.alpha, work=None,batch_size=batch_size),samples_per_epoch=samples_per_epoch, nb_epoch=self.iter)
            self.syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]






if __name__ == "__main__":
    from nltk.corpus import brown, movie_reviews, treebank
    print brown.sents()[0]

    input_file = 'test.txt'
    bk = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),iter=10)
    b = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file))

    print bk.most_similar('the', topn=5)
    print b.most_similar('the', topn=5)

    # br = gensim.models.word2vec.Word2Vec(brown.sents())
    # brk = Word2VecKeras(brown.sents(),iter=10)

    # print brk.most_similar('the', topn=5)
    # print br.most_similar('the', topn=5)

    
