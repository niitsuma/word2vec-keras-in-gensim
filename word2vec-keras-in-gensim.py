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

import numpy as np
#import keras.models.fit_generator
import keras.models
import keras.layers.embeddings
import keras.layers.core
import keras.optimizers
import keras.objectives
import keras.constraints

import KerasLayer.FixedEmbedding


def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None):

    if word not in model.vocab:
        return
    predict_word = model.vocab[word]  # target word (NN output)
    if model.hs:
        y=np.zeros((len(model.vocab)), dtype=REAL)
        for k,i in enumerate(predict_word.code):
            y[predict_word.point[k]]=i
        x0=context_index
        x1=predict_word.index
        return (x0,x1,y)

    # if model.negative:



def train_batch_sg(model, sentences, alpha, work=None):
    train_x0=[]
    train_x1=[]
    train_y=[]
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
                        #yield x0,x1,y
                        train_x0.append([x0])
                        train_x1.append([x1])
                        train_y.append(y)
    train_x=[np.array(train_x0),np.array(train_x1)]
    train_y=np.array(train_y)
    return train_x,train_y

class Word2VecKeras(gensim.models.word2vec.Word2Vec):


    def build_pointmask(self):
        vocab_size = len(self.index2word)


        if self.hs:
            self.pointmask = zeros((len(self.vocab), len(self.vocab)), dtype=REAL)
            for word_index in range(vocab_size):
                for pt in (self.vocab[self.index2word[word_index]]).point :
                    
                    self.pointmask[word_index][pt]=1.0

            
    def build_keras_model(self):

        self.build_pointmask()
        vocab_size=len(self.vocab)

        self.kerasword = keras.models.Sequential()
        self.kerasword.add(keras.layers.embeddings.Embedding(vocab_size,self.vector_size, input_length=1))
        self.kerasword.add(keras.layers.core.Flatten())
        #self.kerasword.add(keras.layers.core.Dense(output_dim=vocab_size,b_constraint = keras.constraints.maxnorm(0),weights=[np.transpose(self.syn1),np.array([0.,0.],'float32')]))
        self.kerasword.add(keras.layers.core.Dense(output_dim=vocab_size,b_constraint = keras.constraints.maxnorm(0)))
        self.kerasword.add(keras.layers.core.Activation('sigmoid'))

        self.keraspointmask = keras.models.Sequential()
        self.keraspointmask.add(KerasLayer.FixedEmbedding.FixedEmbedding(vocab_size,vocab_size, input_length=1,weights=[self.pointmask]))

        self.keraspointmask.add(keras.layers.core.Flatten())

        self.kerasmodel= keras.models.Sequential()
        self.kerasmodel.add(keras.layers.core.Merge([self.kerasword, self.keraspointmask], mode='mul'))

        self.kerasmodel.compile(loss='mse', optimizer='rmsprop')

    def train(self, sentences, total_words=None, word_count=0, chunksize=100, total_examples=None, queue_factor=2, report_delay=1):
        #print 'Word2VecKerastrain'        
        if self.sg:
            self.build_keras_model()
            train_x,train_y=train_batch_sg(self, sentences, self.alpha, work=None)
            #print dir(self.kerasmodel)
            self.kerasmodel.fit(train_x, train_y)
            #self.kerasmodel.fit_generator(train_batch_sg(self, sentences, self.alpha, work=None))
            self.syn0=self.kerasword.layers[0].get_weights()[0]






if __name__ == "__main__":
    from nltk.corpus import brown, movie_reviews, treebank
    print brown.sents()[0]

    input_file = 'test.txt'
    bk = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file))
    b = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file))

    print bk.most_similar('the', topn=5)
    print b.most_similar('the', topn=5)

    # b = Word2VecKeras(brown.sents())  ##memory over flow


    
