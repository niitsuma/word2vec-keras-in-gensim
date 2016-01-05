#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

from numpy import zeros, random, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide

import gensim.models.word2vec
import gensim.models.doc2vec 

from six.moves import xrange, zip
from six import string_types, integer_types, itervalues

import sys
import random

import numpy as np
import operator

import keras.constraints

from keras.utils.np_utils import accuracy
from keras.models import Graph,Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten 
from keras.layers.embeddings import Embedding
#from keras.optimizers import SGD
from keras.objectives import mse

from word2veckeras import train_sg_pair


def train_batch_dbow(model,
                     docs, alpha,
                     work=None,
                     train_words=False, learn_doctags=True, learn_words=True, learn_hidden=True,
                     word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None,
                     batch_size=100):
    #print 'train_batch_dbow'
    batch_count=0
    train_x0=[[0]]*batch_size
    train_x1=[[0]]*batch_size
    train_y=[[0]]*batch_size
        
    while 1:
        for doc in docs:
            for doctag_index in doc.tags:
                for word in doc.words:
                    xy=train_sg_pair(model, word, doctag_index, alpha, learn_vectors=learn_doctags,
                              learn_hidden=learn_hidden, context_vectors=doctag_vectors,
                              context_locks=doctag_locks)
                    if xy !=None:
                        (x0,x1,y)=xy
                        #print xy
                        train_x0[batch_count]=[x0]
                        train_x1[batch_count]=x1
                        train_y[batch_count]=y
                        batch_count += 1
                        if batch_count >= batch_size :
                            yield { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y)}
                            batch_count=0


# def train_batch_dm(model,
#                        docs
#                        #, doc_words
#                        #, doctag_indexes
#                        , alpha, work=None, neu1=None,
#                           learn_doctags=True, learn_words=True, learn_hidden=True,
#                           word_vectors=None, word_locks=None, doctag_vectors=None, doctag_locks=None,
#                        batch_size=100):
#     batch_count=0
#     train_x0=[[0]]*batch_size
#     train_x1=[[0]]*batch_size
#     train_y=[[0]]*batch_size
#     while 1:                      
#         for doc in docs:
#             for doctag_index in doc.tags:
#                 #for word in doc.words:
#                 word_vocabs = [model.vocab[w] for w in doc.words if w in model.vocab and
#                        model.vocab[w].sample_int > model.random.rand() * 2**32]

#                 for pos, word in enumerate(word_vocabs):
#                     reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
#                     start = max(0, pos - model.window + reduced_window)
#                     window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
#                     word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
#                     l1 = np_sum(word_vectors[word2_indexes], axis=0) + np_sum(doctag_vectors[doctag_indexes], axis=0)
#                     count = len(word2_indexes) + len(doctag_indexes)
#                     if model.cbow_mean and count > 1 :
#                         l1 /= count
#                     neu1e = train_cbow_pair(model, word, word2_indexes, l1, alpha,
#                                             learn_vectors=False, learn_hidden=learn_hidden)
#                     if not model.cbow_mean and count > 1:
#                         neu1e /= count
#                     if learn_doctags:
#                         for i in doctag_indexes:
#                             doctag_vectors[i] += neu1e * doctag_locks[i]
#                     if learn_words:
#                         for i in word2_indexes:
#                             word_vectors[i] += neu1e * word_locks[i]

#                 return len(word_vocabs)

                            
                            
class Doc2VecKeras(gensim.models.doc2vec.Doc2Vec):

    def build_keras_model(self):

        vocab_size=len(self.vocab)
        #index_size=vocab_size
        index_size=len(self.docvecs)
        code_dim=vocab_size
        #print 'build_keras_model',vocab_size, index_size, code_dim
        
        self.kerasmodel = Graph()
        self.kerasmodel.add_input(name='point', input_shape=(code_dim,), dtype=REAL)
        self.kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
        self.kerasmodel.add_node(self.kerasmodel.inputs['point'],name='pointnode')
        self.kerasmodel.add_node(Embedding(index_size, self.vector_size, input_length=1),name='embedding', input='index')
        self.kerasmodel.add_node(Flatten(),name='embedflatten',input='embedding')
        self.kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0)), name='sigmoid', input='embedflatten')
        self.kerasmodel.add_output(name='code',inputs=['sigmoid','pointnode'], merge_mode='mul')
        self.kerasmodel.compile('rmsprop', {'code':'mse'})

    def train(self, docs, total_words=None, word_count=0, chunksize=100, total_examples=None, queue_factor=2, report_delay=1):
        #print 'Doc2VecKeras',self.sg
        #batch_size=800 ##optimized 1G mem video card
        #print self.batch_size
        #sys.exit()
        batch_size=3000
        if self.sg:
            #print'Doc2VecKeras',self.sg
            self.build_keras_model()
            samples_per_epoch=int(self.window*2*sum(map(len,docs)))
            #print 'samples_per_epoch',samples_per_epoch

            # gen=train_batch_dbow(self, docs, self.alpha, batch_size=batch_size)
            # print gen
            # for g in gen:
            #     print g
            # sys.exit()
            self.kerasmodel.fit_generator(
                train_batch_dbow(self, docs, self.alpha, batch_size=batch_size)
                ,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter)
            
            self.syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]




class LabeledListSentence(object):
    def __init__(self, words_list):
        """
        words_list like:
        words_list = [
        ['human', 'interface', 'computer'],
        ['survey', 'user', 'computer', 'system', 'response', 'time'],
        ['eps', 'user', 'interface', 'system'],
        ]
        sentence = LabeledListSentence(words_list)
        """
        self.words_list = words_list
    
    def __getitem__(self, index):
        t = [t for t in self]
        return t[index]
    def __iter__(self):
        for i, words in enumerate(self.words_list):
            #yield LabeledSentence(words, ['SENT_{0}'.format(i)])
            yield gensim.models.doc2vec.TaggedDocument(words, [i])


if __name__ == "__main__":

    
    input_file = 'test.txt'
    doc1=gensim.models.doc2vec.TaggedLineDocument(input_file)
    d1=gensim.models.doc2vec.Doc2Vec(doc1,dm=0)
    print(d1.docvecs.most_similar(0))
    #print len(d1.docvecs)
    
    ###debug print
    # print doc1
    # for d in doc1:
    #     print d
    #print doc1.tags
    
    #print d1.docvecs[0]

    #mk1=Doc2VecKeras(doc1,dm=0, batch_size=1000)
    mk1=Doc2VecKeras(doc1,dm=0)    
    print(mk1.docvecs.most_similar(0))
    
    

    
    from nltk.corpus import brown, movie_reviews, treebank
    # print(brown.sents()[0])

    db=gensim.models.doc2vec.Doc2Vec(LabeledListSentence(brown.sents()),dm=0)
    print(db.docvecs.most_similar(0))
    dbk=Doc2VecKeras(LabeledListSentence(brown.sents()),dm=0)
    print(dbk.docvecs.most_similar(0))
    # br = gensim.models.word2vec.Word2Vec(brown.sents())
    # brk = Word2VecKeras(brown.sents(),iter=10)

    # print( brk.most_similar('the', topn=8))
    # print( br.most_similar('the', topn=8))

    
