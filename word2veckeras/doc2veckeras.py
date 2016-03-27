#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

import math
import copy
from Queue import Queue

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
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten, LambdaMerge,Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse

from sklearn.base import BaseEstimator,RegressorMixin, ClassifierMixin
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from word2veckeras import train_sg_pair,train_cbow_pair,queue_to_list,train_prepossess


def train_batch_dbow(model,
                     docs,
                     sub_batch_size=256,batch_size=256
                     ):
    batch_count=0
    sub_batch_count=0
    train_x0 =np.zeros((batch_size,sub_batch_size),dtype='int32')
    train_x1 =np.zeros((batch_size,sub_batch_size),dtype='int32')
    train_y  =np.zeros((batch_size,sub_batch_size),dtype='int8')
    while 1:
        for doc in docs:
            for doctag_index in doc.tags:
                for word in doc.words:
                    xy_gen=train_sg_pair(model, word, doctag_index,)
                    for xy in xy_gen :
                        if xy !=None:
                            (x0,x1,y)=xy
                            train_x0[batch_count][sub_batch_count]=x0
                            train_x1[batch_count][sub_batch_count]=x1
                            train_y[batch_count][sub_batch_count]=y
                            sub_batch_count += 1
                            if sub_batch_count >= sub_batch_size :
                                batch_count += 1
                                sub_batch_count=0
                            if batch_count >= batch_size :
                                yield { 'index':train_x0, 'point':train_x1, 'code':train_y}
                                batch_count=0
                            
def train_batch_dm_xy_generator(model, docs):
    for doc in docs:
        indexed_doctags = model.docvecs.indexed_doctags(doc.tags)
        doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags

        word_vocabs = [model.vocab[w] for w in doc.words if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original doc2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indexes = [word2.index for pos2, word2 in window_pos if pos2 != pos]
                
            xy_gen=train_cbow_pair(model, word, word2_indexes)
            x2=doctag_indexes
            for xy in xy_gen:
                if xy !=None:
                    yield [xy[0],x2,xy[1],xy[2]]

def train_batch_dm(model, docs,batch_size=100,sub_batch_size=1,):
    w_len_queue_dict={}
    w_len_queue=[]
    while 1:
       for xy in train_batch_dm_xy_generator(model,docs):
            if xy != None:
                w_len=len(xy[0])
                if w_len>0:
                    if w_len not in w_len_queue_dict:
                        w_len_queue_dict[w_len]=Queue()
                        w_len_queue.append(w_len)
                    w_len_queue_dict[w_len].put(xy)
            for w_len in w_len_queue:
                if w_len_queue_dict[w_len].qsize() >= batch_size :
                    l=queue_to_list(w_len_queue_dict[w_len],batch_size)
                    train=[[e[i] for e in l] for i in range(4)]
                    yield {'iword':np.array(train[0]),
                           'index':np.array(train[1]),
                            'point':np.array(train[2]),
                            'code':np.array(train[3])}
                    
def train_document_dm_concat_xy_generator(model, docs):
    for doc in docs:
        indexed_doctags = model.docvecs.indexed_doctags(doc.tags)
        doctag_indexes, doctag_vectors, doctag_locks, ignored = indexed_doctags

        word_vocabs = [model.vocab[w] for w in doc.words if w in model.vocab and
                       model.vocab[w].sample_int > model.random.rand() * 2**32]
        null_word = model.vocab['\0']
        pre_pad_count = model.window
        post_pad_count = model.window
        padded_document_indexes = (
            (pre_pad_count * [null_word.index])  # pre-padding
            + [word.index for word in word_vocabs if word is not None]  # elide out-of-Vocabulary words
            + (post_pad_count * [null_word.index])  # post-padding
        )

        for pos in range(pre_pad_count, len(padded_document_indexes) - post_pad_count):
            word_context_indexes = (
                padded_document_indexes[(pos - pre_pad_count): pos]  # preceding words
                + padded_document_indexes[(pos + 1):(pos + 1 + post_pad_count)]  # following words
            )
            predict_word = model.vocab[model.index2word[padded_document_indexes[pos]]]
            xy_gen = train_cbow_pair(model, predict_word, word_context_indexes)
            for xy in xy_gen:
                if xy !=None:
                    x2=doctag_indexes
                    xy1=[xy[0],x2,xy[1],xy[2]]
                    yield xy1
 
def train_document_dm_concat(model, docs,batch_size=100):
    batch_count=0
    train_t=[0]*batch_size

    while 1:
        for xy in train_document_dm_concat_xy_generator(model,docs):
            train_t[batch_count]=xy
            batch_count += 1
            if batch_count >= batch_size :
                train=[[train_t[j][i] for j in range(batch_size) ] for i in range(4)]
                yield {'iword':np.array(train[0]),
                           'index':np.array(train[1]),
                            'point':np.array(train[2]),
                            'code':np.array(train[3])}
                batch_count=0
                      

def build_keras_model_dbow(index_size,vector_size,
                           #vocab_size,
                           context_size,
                           sub_batch_size=1,
                           doctag_vectors=None,
                           hidden_vectors=None,
                           learn_doctags=True,
                           learn_hidden=True,
                           model=None,
                           ):

    """
    >>> index_size=3
    >>> vector_size=2
    >>> context_siz=3
    >>> sub_batch_size=2
    >>> doctag_vectors=np.array([[-1.1,2.2],[-3.2,-4.3],[-1.1,-1.4]],'float32')
    >>> hidden_vectors=np.array([[-1,2],[3,4],[5,6]],'float32')
    >>> kerasmodel=build_keras_model_dbow(index_size=3,vector_size=2,context_size=3,sub_batch_size=2,doctag_vectors=doctag_vectors,hidden_vectors=hidden_vectors)
    >>> ind=[[0,1],[1,0]]
    >>> ipt=[[0,1],[1,2]]
    >>> tmp1=kerasmodel.predict({'index':np.array(ind),'point':np.array(ipt)})['code']
    >>> tmp2=np.array([np.sum(doctag_vectors[ind[i]]*hidden_vectors[ipt[i]], axis=1) for i in range(2)])
    >>> np.linalg.norm(1/(1+np.exp(-tmp2))-tmp1) < 0.001
    True
    """
    
    kerasmodel = Graph()
    kerasmodel.add_input(name='point' , input_shape=(sub_batch_size,), dtype=int)
    kerasmodel.add_input(name='index' , input_shape=(sub_batch_size,), dtype=int)
    if hidden_vectors is None :        
        kerasmodel.add_node(Embedding(context_size, vector_size, input_length=sub_batch_size, ),name='embedpoint', input='point')
    else:
        kerasmodel.add_node(Embedding(context_size, vector_size, input_length=sub_batch_size, weights=[hidden_vectors]),name='embedpoint', input='point')
    if doctag_vectors is None :
        kerasmodel.add_node(Embedding(index_size  , vector_size, input_length=sub_batch_size, ),name='embedindex' , input='index')
    else:
        kerasmodel.add_node(Embedding(index_size  , vector_size, input_length=sub_batch_size, weights=[doctag_vectors]),name='embedindex' , input='index')
    kerasmodel.add_node(Lambda(lambda x:x.sum(2))   , name='merge',inputs=['embedindex','embedpoint'], merge_mode='mul')
    kerasmodel.add_node(Activation('sigmoid'), name='sigmoid', input='merge')
    kerasmodel.add_output(name='code',input='sigmoid')
    kerasmodel.compile('rmsprop', {'code':'mse'})
    return kerasmodel


def build_keras_model_dm(index_size,vector_size,vocab_size,
                         context_size,
                         maxwords,
                         cbow_mean=False,
                         learn_doctags=True, learn_words=True, learn_hidden=True,
                         model=None ,
                         word_vectors=None,doctag_vectors=None,hidden_vectors=None,
                         sub_batch_size=1
                         ):
    """
    >>> word_vectors=np.array([[1,2],[3,4],[5,6]])
    >>> doctag_vectors=np.array([[10,20],[30,40]])
    >>> hidden_vectors=np.array([[1,0],[0,1]])
    >>> sub_batch_size=2
    >>> kerasmodel=build_keras_model_dm(index_size=2,vector_size=2,vocab_size=3,context_size=2,maxwords=2,sub_batch_size=sub_batch_size,word_vectors=word_vectors,doctag_vectors=doctag_vectors,hidden_vectors=hidden_vectors, learn_words=True )
    >>> ind=[[0],[1]]
    >>> iwd=[[1,0],[1,1]]
    >>> ipt=[[1,0],[0,1]]
    >>> tmp1=kerasmodel.predict({'index':np.array(ind),'iword':np.array(iwd),'point':np.array(ipt)})['code']
    >>> tmp2=np.array([ [(word_vectors[iwd[i]].sum(0)+doctag_vectors[i]).dot(hidden_vectors[j]) for j in ipt[i] ] for i in range(2)])
    >>> np.linalg.norm(1/(1+np.exp(-tmp2))-tmp1) < 0.001
    True
    """ 
    kerasmodel = Graph()

    kerasmodel.add_input(name='index',input_shape=(1,)     , dtype=int)
    if doctag_vectors is None :        
        kerasmodel.add_node(Embedding(index_size,   vector_size,trainable=learn_doctags,input_length=1                              ),name='embedindex', input='index')
    else:
        kerasmodel.add_node(Embedding(index_size,   vector_size,trainable=learn_doctags,input_length=1     ,weights=[doctag_vectors]),name='embedindex', input='index')
    kerasmodel.add_input(name='iword',input_shape=(maxwords,), dtype=int)

    if word_vectors is None :
        kerasmodel.add_node(Embedding(vocab_size,   vector_size,trainable=learn_words  ,input_length=maxwords                         ),name='embedword', input='iword')
    else:
        kerasmodel.add_node(Embedding(vocab_size,   vector_size,trainable=learn_words  ,input_length=maxwords,weights=[word_vectors  ]),name='embedword', input='iword')

    kerasmodel.add_input(name='point',input_shape=(sub_batch_size,)     , dtype=int)
    if hidden_vectors is None :
        kerasmodel.add_node(Embedding(context_size, vector_size,trainable=learn_hidden ,input_length=sub_batch_size                          ),name='embedpoint', input='point')
    else:
        kerasmodel.add_node(Embedding(context_size, vector_size,trainable=learn_hidden ,input_length=sub_batch_size  ,weights=[hidden_vectors]),name='embedpoint', input='point')
        
    if cbow_mean:
        kerasmodel.add_node(Lambda(lambda x:x.mean(1),output_shape=(vector_size,)), name='merge',inputs=['embedindex','embedword'], merge_mode='concat', concat_axis=1)
    else:
        kerasmodel.add_node(Lambda(lambda x:x.sum(1),output_shape=(vector_size,)), name='merge',inputs=['embedindex','embedword'], merge_mode='concat', concat_axis=1)

    kerasmodel.add_node(Activation('sigmoid'), name='sigmoid',inputs=['merge','embedpoint'], merge_mode='dot',dot_axes=-1)
    kerasmodel.add_output(name='code',input='sigmoid')
    kerasmodel.compile('rmsprop', {'code':'mse'})
    
    return kerasmodel
    
    
def build_keras_model_dm_concat(index_size,vector_size,vocab_size,
                                #code_dim,
                                context_size,
                                window_size,
                                learn_doctags=True, learn_words=True, learn_hidden=True,
                                model=None ,
                                word_vectors=None,doctag_vectors=None,hidden_vectors=None
                                ):
    """
    >>> syn0=np.array([[1,-2],[-1,2],[2,-2]],'float32')
    >>> word_vectors=syn0
    >>> syn1=np.array([[-1,2,1,-5,4,1,-2,3,-4,5],[3,4,-4,1,-2,6,-7,8,9,1],[5,-6,-8,7,6,-1,2,-3,4,5]],'float32')
    >>> hidden_vectors=syn1
    >>> doctag_vectors=np.array([[-1.1,2.2],[-3.2,-4.3],[-1.1,-1.4]],'float32')
    >>> kerasmodel=build_keras_model_dm_concat(index_size=3,vector_size=2,vocab_size=3,context_size=3,window_size=2,word_vectors=word_vectors,doctag_vectors=doctag_vectors,hidden_vectors=hidden_vectors)
    >>> ind=[[0],[1]]
    >>> iwd=[[0,0,1,2],[1,1,2,0]]
    >>> ipt=[[0],[1]]
    >>> tmp1=kerasmodel.predict({'index':np.array(ind),'iword':np.array(iwd),'point':np.array(ipt)})['code']
    >>> tmp2=np.array([[np.vstack((doctag_vectors[ind[i]],word_vectors[iwd[i]])).flatten().dot(hidden_vectors[j]) for j in ipt[i] ] for i in range(2)])
    >>> np.linalg.norm(1/(1+np.exp(-tmp2))-tmp1) < 0.001
    True
    """

    kerasmodel = Graph()
    kerasmodel.add_input(name='iword' , input_shape=(1,), dtype=int)
    kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
    if word_vectors is None:
        kerasmodel.add_node(Embedding(vocab_size, vector_size,input_length=2*window_size,trainable=learn_words,),name='embedword', input='iword')
    else:
        kerasmodel.add_node(Embedding(vocab_size, vector_size,input_length=2*window_size,trainable=learn_words,weights=[word_vectors]),name='embedword', input='iword')
    if doctag_vectors is None:
        kerasmodel.add_node(Embedding(index_size, vector_size,input_length=1,trainable=learn_doctags,), name='embedindex', input='index')
    else:
        kerasmodel.add_node(Embedding(index_size, vector_size,input_length=1,trainable=learn_doctags,weights=[doctag_vectors]), name='embedindex', input='index')

    kerasmodel.add_input(name='point',input_shape=(1,)     , dtype=int)
    if hidden_vectors is None:
        kerasmodel.add_node(Embedding(context_size, (2*window_size+1)*vector_size,input_length=1, trainable=learn_hidden,),name='embedpoint', input='point')
    else:
        kerasmodel.add_node(Embedding(context_size, (2*window_size+1)*vector_size,input_length=1, trainable=learn_hidden,weights=[hidden_vectors]),name='embedpoint', input='point')

    kerasmodel.add_node(Flatten(),name='merge',inputs=['embedindex','embedword'],merge_mode='concat', concat_axis=1)
    kerasmodel.add_node(Activation('sigmoid'), name='sigmoid',inputs=['merge','embedpoint'], merge_mode='dot',dot_axes=-1)
    kerasmodel.add_output(name='code',input='sigmoid')
    kerasmodel.compile('rmsprop', {'code':'mse'})
    return kerasmodel

        
class Doc2VecKeras(gensim.models.doc2vec.Doc2Vec):
    def train(self, docs=None,
              #batch_size=800,
              learn_doctags=True, learn_words=True, learn_hidden=True,iter=None,
              batch_size=128 #128, #512 #256
              ,sub_batch_size=128 #16 #32 #128 #128  #256 #128 #512 #256 #1
              ):
        
        if iter!=None:
            self.iter=iter
        if docs==None:
            docs=self.docvecs
            
        train_prepossess(self)
        
        # if self.negative>0 and self.hs :
        #     self.keras_context_negative_base_index=len(self.vocab)
        #     self.keras_context_index_size=len(self.vocab)*2
        #     self.keras_syn1=np.vstack((self.syn1,self.syn1neg))
        # else:
        #     self.keras_context_negative_base_index=0
        #     self.keras_context_index_size=len(self.vocab)
        #     if self.hs :
        #         self.keras_syn1=self.syn1
        #     else:
        #         self.keras_syn1=self.syn1neg

        # self.neg_labels = []
        # if self.negative > 0:
        #     # precompute negative labels optimization for pure-python training
        #     self.neg_labels = np.zeros(self.negative + 1,dtype='int8')
        #     self.neg_labels[0] = 1
        

        # word_context_size_max=0
        # if self.hs :
        #     word_context_size_max += max(len(self.vocab[w].point) for w in self.vocab if hasattr(self.vocab[w],'point'))
        # if self.negative > 0:
        #     word_context_size_max += self.negative + 1

        vocab_size=len(self.vocab)
        index_size=len(self.docvecs)

            
        self.batch_size=batch_size
        batch_size=batch_size
        if self.sg:
            samples_per_epoch=max(1,int((self.word_context_size_max*self.window*2*sum(map(len,docs)))/(sub_batch_size)))
            self.kerasmodel=build_keras_model_dbow(index_size=index_size,vector_size=self.vector_size,
                                                   context_size=self.keras_context_index_size,
                                                   model=self,
                                                   learn_doctags=learn_doctags,
                                                   learn_hidden=learn_hidden,
                                                   hidden_vectors=self.keras_syn1,
                                                   doctag_vectors=self.docvecs.doctag_syn0,
                                                   sub_batch_size=sub_batch_size
                                                   )
            gen=train_batch_dbow(self, docs, sub_batch_size=sub_batch_size,batch_size=batch_size)
            self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)
        else:
            if self.dm_concat:
                samples_per_epoch=int(self.word_context_size_max*sum(map(len,docs)))
                self.kerasmodel=build_keras_model_dm_concat(index_size,self.vector_size,vocab_size,
                                                            context_size=self.keras_context_index_size,
                                                            window_size=self.window,
                                                            model=self,
                                                            learn_doctags=learn_doctags, learn_words=learn_words, learn_hidden=learn_hidden,
                                                            word_vectors=self.syn0,
                                                            hidden_vectors=self.keras_syn1,
                                                            doctag_vectors=self.docvecs.doctag_syn0
                                                            )
                gen= train_document_dm_concat(self, docs, batch_size=batch_size)
                self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter, verbose=0)
                self.syn0=self.kerasmodel.nodes['embedword'].get_weights()[0]
            else:
                samples_per_epoch=int(self.word_context_size_max*sum(map(len,docs)))
                self.kerasmodel=build_keras_model_dm(index_size,self.vector_size,vocab_size,
                                                     self.keras_context_index_size,
                                                     maxwords=self.window*2+1,
                                                     model=self,
                                                     learn_doctags=learn_doctags, learn_words=learn_words, learn_hidden=learn_hidden,
                                                     word_vectors=self.syn0,
                                                     doctag_vectors=self.docvecs.doctag_syn0,
                                                     hidden_vectors=self.keras_syn1,
                                                     cbow_mean=self.cbow_mean
                                                     )

                gen=train_batch_dm(self, docs, batch_size=batch_size)
                self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)
                self.syn0=self.kerasmodel.nodes['embedword'].get_weights()[0]
        self.docvecs.doctag_syn0=self.kerasmodel.nodes['embedindex'].get_weights()[0]
        if self.negative>0 and self.hs :
            syn1tmp=self.kerasmodel.nodes['embedpoint'].get_weights()[0]
            self.syn1=syn1tmp[0:len(self.vocab)]
            self.syn1neg=syn1tmp[len(self.vocab):2*len(self.vocab)]
        elif self.hs:
            self.syn1=self.kerasmodel.nodes['embedpoint'].get_weights()[0]
        else:
            self.syn1neg=self.kerasmodel.nodes['embedpoint'].get_weights()[0]

    # def infer_vector_keras(self, doc_words, steps=10):
    #     vocab_size=len(self.vocab)
    #     docs=LabeledListSentence([doc_words])
    #     batch_size=5
    #     #batch_size=10
    #     #batch_size=100
    #     #batch_size=1000
    #     samples_per_epoch=int(self.window*2*sum(map(len,docs)))
        
    #     count_max= steps * samples_per_epoch/batch_size +steps
    #     #print 'count_max',count_max
    #     # print self.kerasmodel_infer.nodes['embedword'].get_weights()
    #     # print self.kerasmodel_infer.nodes[ 'sigmoid'].get_weights()

    #     doctag_vectors = empty((1, self.vector_size), dtype=REAL)
    #     doctag_vectors[0] = self.seeded_vector(' '.join(doc_words))
    #     doctag_locks = ones(1, dtype=REAL)
    #     doctag_indexes = [0]
    #     #self.kerasmodel_infer.nodes.get_weights()
    #     self.kerasmodel_infer.nodes['embedindex'].set_weights([ doctag_vectors])
    #     count =0
    #     if self.sg:
    #         for g in train_batch_dbow(self, docs, self.alpha, batch_size=batch_size):
    #             self.kerasmodel_infer.fit(g, nb_epoch=1, verbose=0)
    #             count +=1
    #             if count > count_max:
    #                 break
    #     elif self.dm_concat:
    #         for g in train_document_dm_concat(self, docs, batch_size=batch_size):
    #             self.kerasmodel_infer.fit(g, nb_epoch=1, verbose=0)
    #             count +=1
    #             if count > count_max:
    #                 break
    #     else:
    #         for g in train_batch_dm(self, docs, batch_size=batch_size):
    #             self.kerasmodel_infer.fit(g, nb_epoch=1, verbose=0)
    #             count +=1
    #             if count > count_max:
    #                 break
    #     vecs=self.kerasmodel_infer.nodes['embedindex'].get_weights()[0]
    #     return vecs[0]

                
    def train_with_word2vec_instance(self,docs,w2v,dm=None, **kwargs):
        if self.dm_concat and not w2v.null_word :
            raise ValueError("self.dm_concat=1 need Word2Vec(null_word=1)") #KeyError ?
        
        if dm == None :
            #if not self.dm_concat:
            self.sg = w2v.sg
        else:
            self.sg=(1+dm) % 2
            
        self.window = w2v.window 
        self.min_count =w2v.min_count
        self.sample =w2v.sample
        self.cbow_mean=w2v.cbow_mean
        
        self.negative = w2v.negative
        self.hs=w2v.hs
            
        #self.alpha = w2v.alpha 

        self.vector_size=w2v.vector_size
        if not self.dm_concat:
            self.layer1_size= w2v.layer1_size


        self.raw_vocab=w2v.raw_vocab
        self.index2word=w2v.index2word
        self.sorted_vocab = w2v.sorted_vocab

        self.vocab=w2v.vocab

        self.max_vocab_size = w2v.max_vocab_size

        
        #self.build_vocab(docs)
        for document_no, document in enumerate(docs):
            document_length = len(document.words)
            for tag in document.tags:
                self.docvecs.note_doctag(tag, document_no, document_length)
        self.reset_weights()


        self.syn0=w2v.syn0

        if w2v.hs:
            if not self.dm_concat:
                self.syn1=w2v.syn1
        if w2v.negative:
            if not self.dm_concat:
                self.syn1neg=w2v.syn1neg
            self.cum_table=w2v.cum_table

        
        self.train(docs,**kwargs)
        #self.train(docs,learn_words=learn_words,**kwargs)
        

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

class SentenceClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 sents_shuffle=False,
                 doc2vec=gensim.models.doc2vec.Doc2Vec()
                 ):
        argdict= locals()
        argdict.pop('argdict',None)
        argdict.pop('self',None)
        vars(self).update(argdict)
        #print argdict
    
    def fit(self, X, y):
        self.sents_train=X
        self.Y_train=y
        return self
    
    def doc2vec_set(self,all_docs):
        #print 'doc2vec_set,SentenceClassifier'
        if hasattr(self.doc2vec, 'syn0'):
            self.doc2vec.reset_weights()
            #del self.doc2vec.syn0
            delattr(self.doc2vec, 'syn0')
        self.doc2vec.build_vocab(all_docs)
        self.doc2vec.train(all_docs)

    def predict(self,X):
        self.sents_test=X
        self.sents_all=self.sents_train + self.sents_test

        if self.sents_shuffle :
            s_indexs=range(len(self.sents_all))
            random.shuffle(s_indexs)
            s_invers_indexs=range(len(s_indexs))
            for n in range(len(s_indexs)):
                s_invers_indexs[s_indexs[n]]=n
            sents_all=[self.sents_all[n] for n in s_indexs]
        else:
            sents_all=self.sents_all
        all_docs = list(LabeledListSentence(self.sents_all))
        
        self.doc2vec_set(all_docs)
        #print 'size',self.doc2vec.vector_size

        self.X_train= [self.doc2vec.infer_vector(s) for s in self.sents_train]
        self.X_test= [self.doc2vec.infer_vector(s) for s in self.sents_test]
        self.logistic =LogisticRegressionCV(class_weight='balanced')#,n_jobs=-1)
        self.logistic.fit(self.X_train,self.Y_train)
        Y_test_predict=self.logistic.predict(self.X_test)
        return Y_test_predict

doc2vec_init_param_dict={
    #'sents_shuffle': False,
    'comment': None,
    'dm': 1, 'dm_mean': 0, 'hs': 1, 'sample': 0, 'seed': 1, 'dbow_words': 0, 'dm_concat': 0,
    'min_count': 5, 'max_vocab_size': None, 'alpha': 0.025, 'dm_tag_count': 1,
    'docvecs_mapfile': None,
    'size': 300,
    'documents': None, 'trim_rule': None, 'workers': 1, 'negative': 0, 'docvecs': None, 'window': 8,
    #'kwargs': {},
    'min_alpha': 0.0001,
    'iter':1
    }

class Doc2VecClassifier(SentenceClassifier):
    def __init__(self,
                 sents_shuffle=False,
                 documents=None, size=300, alpha=0.025, window=8, min_count=5,
                 max_vocab_size=None, sample=0, seed=1, workers=1, min_alpha=0.0001,
                 dm=1, hs=1, negative=0, dbow_words=0, dm_mean=0, dm_concat=0, dm_tag_count=1,
                 docvecs=None, docvecs_mapfile=None, comment=None, trim_rule=None,
                 iter=1,
                 #**kwargs
                 ):
        argdict= locals()
        argdict.pop('argdict',None)
        argdict.pop('self',None)
        vars(self).update(doc2vec_init_param_dict)
        vars(self).update(argdict)
        
    def doc2vec_set(self,all_docs):
        #print 'doc2vec_set,Doc2VecClassifier'
        doc2vec_init_dict={k:vars(self)[k] for k in vars(self).keys()  if k in doc2vec_init_param_dict}
        doc2vec_init_dict['documents']=all_docs
        #print doc2vec_init_dict
        self.doc2vec=gensim.models.doc2vec.Doc2Vec(**doc2vec_init_dict)
            
    
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    
    input_file = 'test.txt'
    doc1=gensim.models.doc2vec.TaggedLineDocument(input_file)
    for d in doc1:
        doc_words1=d.words
        break;
    d_size=5
    dv1=Doc2VecKeras(                doc1,size=d_size,dm=0,dm_concat=1,hs=0,negative=5,iter=1)
    dvk1=gensim.models.doc2vec.Doc2Vec(doc1,size=d_size,dm=0,dm_concat=1,hs=0,negative=5,iter=1)

    print dv1.docvecs[0]
    print dvk1.docvecs[0]
    
