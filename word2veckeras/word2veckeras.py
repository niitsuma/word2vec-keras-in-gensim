#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

import math
from Queue import Queue

from numpy import zeros, random, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide

import gensim.models.word2vec 

from six.moves import xrange, zip
from six import string_types, integer_types, itervalues

import sys
import random

import numpy as np
import copy

import keras.constraints

from keras.utils.np_utils import accuracy
from keras.models import Graph,Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten , Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse


def queue_to_list(q,extract_size):
    """ Dump a Queue to a list """
    # A new list
    l = []
    count=0
    while q.qsize() > 0:
        count +=1
        if count >extract_size:
            break
        l.append(q.get())

    return l



def train_sg_pair(model, word, context_index, alpha=None, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None,
                  scale=1
                  ):

    if word not in model.vocab:
        #yield None
        return
    predict_word = model.vocab[word]  # target word (NN output)

    for i,p in enumerate(predict_word.point):
        #print p
        yield context_index,p,predict_word.code[i]
    #     #sys.exit()
    # #if model.hs:
    #     y=np.zeros((len(model.vocab)), dtype=REAL)
    #     x1=np.zeros((len(model.vocab)), dtype=REAL)
    #     x1[predict_word.point]=1 #*scale
    #     y[predict_word.point]=predict_word.code
    #     x0=context_index
    #     #x1=predict_word.index
    #     return x0,x1,y

    # if model.negative:
    #     x0=context_index
    #     y=np.zeros((len(model.vocab)), dtype=REAL)
    #     x1=np.zeros((len(model.vocab)), dtype=REAL)

        
    #     word_indices = [predict_word.index]
    #     while len(word_indices) < model.negative + 1:
    #         w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
    #         if w != predict_word.index:
    #             word_indices.append(w)

    #     x1[word_indices]=1
    #     y[word_indices]=model.neg_labels
    #     return x0,x1,y ##missed in develop branch

def train_batch_sg(model, sentences, alpha=None, work=None,sub_batch_size=256,batch_size=256):
    
    batch_count=0
    sub_batch_count=0
    train_x0 =np.zeros((batch_size,sub_batch_size),dtype='int32')
    train_x1 =np.zeros((batch_size,sub_batch_size),dtype='int32')
    train_y  =np.zeros((batch_size,sub_batch_size),dtype='int8')

    while 1:
    #if 1 :        
        #print 'train_batch_sg'
        for sentence in sentences:
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                #window_length=len(word_vocabs[start:(pos + model.window + 1 - reduced_window)])
                #print window_length,
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        xy_gen=train_sg_pair(model, model.index2word[word.index], word2.index)
                        for xy in xy_gen :
                            if xy !=None:
                                (x0,x1,y)=xy
                                # train_x0[batch_count]=[x0]
                                # train_x1[batch_count]=[x1]
                                # train_y[batch_count]=[y]
                                train_x0[batch_count][sub_batch_count]=x0
                                train_x1[batch_count][sub_batch_count]=x1
                                train_y[batch_count][sub_batch_count]=y
                                sub_batch_count += 1
                                if sub_batch_count >= sub_batch_size :
                                    batch_count += 1
                                    sub_batch_count=0
                                if batch_count >= batch_size :
                                    #yield { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y)}
                                    yield { 'index':train_x0, 'point':train_x1, 'code':train_y}
                                    batch_count=0
        # yield { 'index':np.array(train_x0[:batch_count+1]), 'point':np.array(train_x1[:batch_count+1]), 'code':np.array(train_y[:batch_count+1])}
        # batch_count=0
        

def build_keras_model_sg(index_size,vector_size,vocab_size,code_dim,sub_batch_size=256,learn_vectors=True,learn_hidden=True,model=None):

    #print 'build_keras_model_sg'

    kerasmodel = Graph()
    
    # kerasmodel.add_input(name='point' , input_shape=(sub_batch_size,), dtype=int)
    # kerasmodel.add_input(name='index' , input_shape=(sub_batch_size,), dtype=int)
    # kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1,weights=[model.syn0]),name='embedding', input='index')
    # kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1,weights=[model.syn1]),name='embedpoint', input='point')

    
    kerasmodel.add_input(name='point' , input_shape=(1,), dtype=int)
    kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
    kerasmodel.add_node(Embedding(index_size, vector_size, input_length=sub_batch_size,weights=[model.syn0]),name='embedding', input='index')
    kerasmodel.add_node(Embedding(index_size, vector_size, input_length=sub_batch_size,weights=[model.syn1]),name='embedpoint', input='point')
    
    #kerasmodel.add_node(Embedding(index_size, vector_size, weights=[model.syn1]),name='embedpoint', input='point')
    #kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1),name='embedpoint', input='point')
    #kerasmodel.add_node(Merge([kerasmodel.nodes['embedword'],kerasmodel.nodes['embedpoint'] ],mode='mul'),name='merge')
    #kerasmodel.add_node(Activation('sigmoid'), name='sigmoid', input='merge')
    #kerasmodel.add_node(Activation('sigmoid'), name='sigmoid',inputs=['embedding','embedpoint'], merge_mode='dot',dot_axes=2)
    #kerasmodel.add_node(Activation('sigmoid'), name='sigmoid',inputs=['embedding','embedpoint'], merge_mode='dot',dot_axes=-1)
    #kerasmodel.add_node(Lambda(lambda x:x.mean(1),output_shape=(sub_batch_size,)) , name='merge',inputs=['embedding','embedpoint'], merge_mode='dot')
    #kerasmodel.add_node(Activation('sigmoid'), name='sigmoid', input='merge')

    kerasmodel.add_node(Lambda(lambda x:x.sum(2))   , name='merge',inputs=['embedding','embedpoint'], merge_mode='mul')
    kerasmodel.add_node(Activation('sigmoid'), name='sigmoid', input='merge')

    
    #kerasmodel.add_node(Lambda(lambda x:x), name='sigmoid',inputs=['embedword','embedpoint'], merge_mode='mul')
    #kerasmodel.add_node(Flatten(),name='embedflatten',input='embedding')
    #kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0),weights=[model.syn1.T,np.zeros((code_dim))]), name='sigmoid', input='embedflatten')
    #kerasmodel.add_output(name='code',inputs=['sigmoid','pointnode'], merge_mode='mul')
    
    kerasmodel.add_output(name='code',input='sigmoid')
    #kerasmodel.compile(loss='binary_crossentropy', optimizer='rmsprop')
    
    #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    #kerasmodel.compile(loss='mean_squared_error', optimizer=sgd, class_mode="binary")
    #kerasmodel.compile(sgd,{'code':'mse'}) 
    #kerasmodel.compile(loss='binary_crossentropy', optimizer=sgd)

    # rmsprop1=keras.optimizers.RMSprop(lr=0.0001)
    #rmsprop1=keras.optimizers.RMSprop(lr=1.0)
    #kerasmodel.compile(rmsprop1, {'code':'mse'})
    # kerasmodel.compile(sgd, {'code':'mse'})
    #kerasmodel.compile(sgd, {'code':'binary_crossentropy'})
  

    kerasmodel.compile('rmsprop', {'code':'mse'})
    #kerasmodel.compile('sgd', {'code':'mse'})
    return kerasmodel



def train_cbow_pair(model, word, input_word_indices, l=None, alpha=None, learn_vectors=True, learn_hidden=True):

    for i,p in enumerate(word.point):
        #print p
        yield input_word_indices,[p],[word.code[i]]
    # if model.hs:
    #     x0=input_word_indices
    #     x1=np.zeros((len(model.vocab)), dtype=REAL)
    #     x1[word.point]=1
    #     y=np.zeros((len(model.vocab)), dtype=REAL)
    #     y[word.point]=word.code
    #     return x0,x1,y
    
    # if model.negative:
    #     word_indices = [word.index]
    #     while len(word_indices) < model.negative + 1:
    #         w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
    #         if w != word.index:
    #             word_indices.append(w)
    #     x0=input_word_indices
    #     x1=np.zeros((len(model.vocab)), dtype=REAL)
    #     x1[word_indices]=1
    #     y=np.zeros((len(model.vocab)), dtype=REAL)
    #     y[word_indices]=model.neg_labels
    #     return x0,x1,y
        


def train_batch_cbow_xy_generator(model, sentences):
    for sentence in sentences:
        word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and  model.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, word in enumerate(word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
            xy_gen=train_cbow_pair(model, word , word2_indices , None, None)
            for xy in xy_gen:
                if xy !=None:
                    yield xy

def train_batch_cbow(model, sentences, alpha=None, work=None, neu1=None,batch_size=256):
    w_len_queue_dict={}
    w_len_queue=[]

    while 1:
        print 'train_batch_cbow'
        for xy in train_batch_cbow_xy_generator(model, sentences):
            if xy != None:
                w_len=len(xy[0])
                if w_len>0:
                    if w_len not in w_len_queue_dict:
                        w_len_queue_dict[w_len]=Queue()
                        w_len_queue.append(w_len)
                    w_len_queue_dict[w_len].put(xy)
            for w_len in w_len_queue:
                #print w_len,w_len_queue_dict[w_len]._qsize()
                if w_len_queue_dict[w_len].qsize() >= batch_size :
                    #print w_len_queue_dict[w_len]
                    l=queue_to_list(w_len_queue_dict[w_len],batch_size)
                    #train=zip(*l)
                    train=[[e[i] for e in l] for i in range(3)]
                    yield { 'index':np.array(train[0]),
                            'point':np.array(train[1]),
                            'code':np.array(train[2])}

        
def build_keras_model_cbow(index_size,vector_size,vocab_size,code_dim,sub_batch_size=1,model=None,cbow_mean=False):
 
    kerasmodel = Graph()
    kerasmodel.add_input(name='point' , input_shape=(1,), dtype=int)
    kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
    kerasmodel.add_node(Embedding(index_size, vector_size, weights=[model.syn0]),name='embedding', input='index')
    kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1, weights=[model.syn1]),name='embedpoint', input='point')
    if cbow_mean:
        kerasmodel.add_node(Lambda(lambda x:x.mean(1),output_shape=(vector_size,)),name='average',input='embedding')
    else:
        kerasmodel.add_node(Lambda(lambda x:x.sum(1),output_shape=(vector_size,)),name='average',input='embedding')
    
    kerasmodel.add_node(Activation('sigmoid'), name='sigmoid',inputs=['average','embedpoint'], merge_mode='dot',dot_axes=-1)
    #kerasmodel.add_node(Lambda(lambda x:x), name='sigmoid',inputs=['average','embedpoint'], merge_mode='dot',dot_axes=-1)

    kerasmodel.add_output(name='code',input='sigmoid')
    kerasmodel.compile('rmsprop', {'code':'mse'})

    
    # kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
    # kerasmodel.add_input(name='point', input_shape=(code_dim,), dtype=REAL)
    # kerasmodel.add_node(kerasmodel.inputs['point'],name='pointnode')

    # kerasmodel.add_node(Embedding(index_size, vector_size,weights=[model.syn0]),name='embedding', input='index')
    # if cbow_mean:
    #     kerasmodel.add_node(Lambda(lambda x:x.mean(1),output_shape=(vector_size,)),name='average',input='embedding')
    # else:
    #     kerasmodel.add_node(Lambda(lambda x:x.sum(1),output_shape=(vector_size,)),name='average',input='embedding')
    # kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0),weights=[model.syn1.T,np.zeros((code_dim))]), name='sigmoid', input='average')
    # kerasmodel.add_output(name='code',inputs=['sigmoid','pointnode'], merge_mode='mul')
    # kerasmodel.compile('rmsprop', {'code':'mse'})
    
    return kerasmodel

def copy_word2vec_instance_from_to(w2v,w2v_to,sentences=None,documents=None):# ,dm=None, **kwargs):
    
        # if w2v_to.dm_concat and not w2v.null_word :
        #     raise ValueError("w2v_to.dm_concat=1 need Word2Vec(null_word=1)") #KeyError ?

        if hasattr(w2v,'dm'):
            if w2v.dm is None :
            #if not w2v_to.dm_concat:
                w2v_to.sg = w2v.sg
            else:
                w2v_to.sg=(1+w2v.dm) % 2
        else:
            w2v_to.sg = w2v.sg
                
        w2v_to.window = w2v.window 
        w2v_to.min_count =w2v.min_count
        w2v_to.sample =w2v.sample
        w2v_to.cbow_mean=w2v.cbow_mean
        
        w2v_to.negative = w2v.negative
        w2v_to.hs=w2v.hs
            
        w2v_to.alpha = w2v.alpha 

        w2v_to.vector_size=w2v.vector_size
        
        if hasattr(w2v,'dm_concat') and hasattr(w2v_to,'dm_concat'):
            if not w2v_to.dm_concat:
                w2v_to.layer1_size= w2v.layer1_size


        w2v_to.raw_vocab=w2v.raw_vocab
        w2v_to.index2word=w2v.index2word
        w2v_to.sorted_vocab = w2v.sorted_vocab

        w2v_to.vocab=w2v.vocab

        w2v_to.max_vocab_size = w2v.max_vocab_size

        if hasattr(w2v,'dm'):
            docs=documents
            #w2v_to.build_vocab(docs)
            for document_no, document in enumerate(docs):
                document_length = len(document.words)
                for tag in document.tags:
                    w2v_to.docvecs.note_doctag(tag, document_no, document_length)
        w2v_to.reset_weights()

        w2v_to.syn0=w2v.syn0

        if w2v.hs:
            #if not w2v_to.dm_concat:
            w2v_to.syn1=w2v.syn1
        if w2v.negative:
            #if not w2v_to.dm_concat:
            w2v_to.syn1neg=w2v.syn1neg
            w2v_to.cum_table=w2v.cum_table
            
        return w2v_to
        #w2v_to.train(docs,**kwargs)
        #self.train(docs,learn_words=learn_words,**kwargs)


class Word2VecKeras(gensim.models.word2vec.Word2Vec):

    def compare_w2v(self,w2v2):
        return np.mean([np.linalg.norm(self[w]-w2v2[w]) for w in self.vocab if w in w2v2.vocab])

    def train(self, sentences, total_words=None, word_count=0,
               total_examples=None, queue_factor=2, report_delay=1,
               batch_size=128 #512 #256
               ,sub_batch_size=16 #32 #128 #128  #256 #128 #512 #256 #1
              ):
        #print 'Word2VecKerastrain'
        if self.hs and self.negative>0 :
            raise ValueError("both using hs and negative not implemented")
        
        trim_rule=None
        if len(self.vocab) == 0 : #not hasattr(self, 'syn0'):
            #print 'build_vocab'
            self.build_vocab(sentences, trim_rule=trim_rule)
            #print self.syn0
        vocab_size=len(self.vocab)

        sub_batch_size_update=False
        if hasattr(self,'sub_batch_size'):
            if self.sub_batch_size != sub_batch_size :
                sub_batch_size_update=True
                self.sub_batch_size=sub_batch_size
        if self.sg:
            samples_per_epoch=max(1,int((self.iter*self.window*2*sum(map(len,sentences)))/(sub_batch_size)))
            
            # sub_batch_size=2
            # batch_size=3
            # samples_per_epoch=4
            
            #print 'samples_per_epoch',samples_per_epoch,batch_size,sub_batch_size
            
            if not hasattr(self, 'kerasmodel') or sub_batch_size_update:
                self.kerasmodel=build_keras_model_sg(index_size=vocab_size,vector_size=self.vector_size,vocab_size=vocab_size,code_dim=vocab_size,sub_batch_size=sub_batch_size,model=self)
                
            #print self.kerasmodel.predict({'index':np.array([[0,0],[1,0],[1,1]]),'point':np.array([[0,1],[1,0],[2,1]])})
            #self.kerasmodel.fit({'index':np.array([[0,0],[1,0],[1,1]]),'point':np.array([[0,1],[1,0],[2,1]]),'code':np.array([[0,1],[1,1],[1,1]])}, nb_epoch=1, verbose=1)

            #print self.kerasmodel.predict({'index':np.array([[0],[1]]),'point':np.array([[1],[0]])})
            #sys.exit()

            #self.batch_size=batch_size
            gen=train_batch_sg(self, sentences, sub_batch_size=sub_batch_size,batch_size=batch_size)
            
            #wv0=np.copy(self.kerasmodel.nodes['embedding'].get_weights()[0][0])
            #wv0=self.syn0
            self.kerasmodel.nodes['embedding'].set_weights([np.copy(self.syn0)])
            self.kerasmodel.nodes['embedpoint'].set_weights([np.copy(self.syn1)])
            self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter, verbose=0)
            
            # for n in range(self.iter):
            #     #count =0
            #     for g in gen:
            #         print g
            #         self.kerasmodel.fit(g, nb_epoch=1, verbose=1)
            #         sys.exit()
            #         #count +=1
            #         # if count > self.iter * samples_per_epoch/batch_size :
            #         #     break
            #     #print count

            #print self.kerasmodel.nodes['embedding'].get_weights()[0][0]
            #sys.exit()
            
            #super(Doc2Vec, self).reset_from(other_model)
  
            self.syn0=np.copy(self.kerasmodel.nodes['embedding'].get_weights()[0])
            self.syn1=np.copy(self.kerasmodel.nodes['embedpoint'].get_weights()[0])
            #super(Word2VecKeras,self).syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]
            #print 'inner1',np.linalg.norm(wv0-self.syn0)
            #super(Word2VecKeras,self).reset_from(self)
            #print wv0
            #print 'inner2',np.linalg.norm(wv0-self.syn0)
            
            #sys.exit()

        else:
            samples_per_epoch=int(sum(map(len,sentences)))
            #samples_per_epoch=max(1,int(self.iter*self.window*2*sum(map(len,sentences))/sub_batch_size))
            if not hasattr(self, 'kerasmodel'):
                self.kerasmodel=build_keras_model_cbow(index_size=vocab_size,vector_size=self.vector_size,vocab_size=vocab_size,code_dim=vocab_size,model=self,cbow_mean=self.cbow_mean)

            #sys.exit()

            #wv0=copy.copy(self.kerasmodel.nodes['embedding'].get_weights()[0][0])
            self.kerasmodel.fit_generator(train_batch_cbow(self, sentences, self.alpha, work=None,batch_size=batch_size),samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)

            #count =0
            # for g in train_batch_cbow(self, sentences, self.alpha, work=None,batch_size=batch_size):
            #     self.kerasmodel.fit(g, nb_epoch=1, verbose=0)
            #     count +=1
            #     if count > self.iter * samples_per_epoch/batch_size :
            #         break

            #print wv0
            #print self.kerasmodel.nodes['embedding'].get_weights()[0][0]
            #sys.exit()
            self.syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]


if __name__ == "__main__":

    def compare_w2v(w2v1,w2v2):
        return np.mean([np.linalg.norm(w2v1[w]-w2v2[w]) for w in w2v1.vocab if w in w2v2.vocab])
    
    input_file = 'test.txt'
    
    v_iter=1
    v_size=5
    sg_v=1
    topn=4
    sents=gensim.models.word2vec.LineSentence(input_file)
    
    # vsk = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),iter=v_iter)
    # sys.exit()
    
    # vs = gensim.models.word2vec.Word2Vec(sents)
    # print( vsk.most_similar('the', topn=5))
    # print( vs.most_similar('the', topn=5))
    
    # vck = Word2VecKeras(sents,sg=0,iter=v_iter)
    # vc = gensim.models.word2vec.Word2Vec(sents,sg=0)
    # print( vck.most_similar('the', topn=5))
    # print( vc.most_similar('the', topn=5))

    vs1 = gensim.models.word2vec.Word2Vec(sents,sg=sg_v,size=v_size,iter=1)
    # # #vsk1 = Word2VecKeras(sents,sg=sg_v,size=v_size,iter=1)
    #print( vs1.most_similar('the', topn=topn))
    vsk1 = Word2VecKeras(sents,sg=sg_v,size=v_size,iter=1)
    #print( vsk1.most_similar('the', topn=topn))
    print 'compare',vsk1.compare_w2v(vs1)
    #print vsk1['the']
    vsk1.iter=20
    vsk1.train(sents,batch_size=100,sub_batch_size=64)
    #print( vsk1.most_similar('the', topn=topn))
    print 'compare',vsk1.compare_w2v(vs1)
    #print vsk1['the']
    
    print vs1['the']
    print vsk1['the']
    
    #sys.exit()
    
    # vsk1.iter=3
    # vsk1.train(gensim.models.word2vec.LineSentence(input_file),batch_size=2,sub_batch_size=1)
    # print( vsk1.most_similar('the', topn=topn))
    # print vsk1['the']

    
    # vck1 = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),sg=0,size=5,iter=v_iter)
    # vc1 = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file),sg=0,size=5,iter=3)
    # print vck1['the']
    # print vc1['the']

    #sys.exit()

    ## negative sampling has bug
    
    # vsnk = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),iter=3,hs=0,negative=5)
    # vsn = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file),negative=5)
    # print( vsnk.most_similar('the', topn=5))
    # print( vsn.most_similar('the', topn=5))
    
    # vcnk = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),iter=3,sg=0,hs=0,negative=5)
    # vcn = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file),sg=0,negative=5)
    # print( vcnk.most_similar('the', topn=5))
    # print( vcn.most_similar('the', topn=5))


    #sys.exit()

    
    
    from nltk.corpus import brown #, movie_reviews, treebank
    #print(brown.sents()[0])
    #brown_sents=list(brown.sents())
    #brown_sents=list(brown.sents()[:10000])
    brown_sents=list(brown.sents())[:2000]
    v_iter=2



    br = gensim.models.word2vec.Word2Vec(brown_sents,sg=sg_v,iter=1)
    print br.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=topn)
    brk =Word2VecKeras(brown_sents,sg=sg_v,iter=1)
    print brk.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=topn)
    print 'compare',brk.compare_w2v(br)
    #print(brk.most_similar('the', topn=5))
    #print brk['the']
    brk.train(brown_sents)
    print 'compare',brk.compare_w2v(br)
    #print brk.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=topn)
    #print brk.most_similar('the', topn=5)
    #print brk['the']

    br_dummy = gensim.models.word2vec.Word2Vec(brown_sents,sg=sg_v,iter=1)
    copy_word2vec_instance_from_to(brk,br_dummy)
    print br_dummy.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=topn)
    #print(br_dummy.most_similar('the', topn=5))
    

    sys.exit()

    #ns=[200,400,1000]
    ns=[1,2,20,100,200,400,1000]
    # ns=[1,2,20]
    # br = gensim.models.word2vec.Word2Vec(brown_sents,sg=sg_v,iter=1)
    # for n in ns :
    #     #print n
    #     br.iter=n
    #     br.train(brown_sents)#,sg=sg_v,iter=n)
    #     print n,compare_w2v(br,brk)
    #     print br.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=topn)

        #print br['the']
    #sys.exit()
    
    for n in ns :
        brk.iter=n
        #brck =Word2VecKeras(brown_sents,iter=n,sg=sg_v)
        brk.train(brown_sents)#,iter=n,sg=sg_v)
        print n,compare_w2v(brk,br)
        #print brk.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=topn)
        copy_word2vec_instance_from_to(brk,br_dummy)
        print br_dummy.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=topn)

        #print brck['the']

    sys.exit()
        
    print( brc.most_similar('the', topn=5))
    print( brck.most_similar('the', topn=5))
    sys.exit()


    br1 = gensim.models.word2vec.Word2Vec(brown_sents,size=5,iter=v_iter)
    brk1 = Word2VecKeras(brown_sents,size=5,iter=3)
    print( brk1['the'])
    print( br1['the'])


    brc1 = gensim.models.word2vec.Word2Vec(brown_sents,sg=0,size=5,iter=v_iter)
    brck1 = Word2VecKeras(brown_sents,sg=0,size=5,iter=3)
    print( brck1['the'])
    print( brc1['the'])



    
    # brn = gensim.models.word2vec.Word2Vec(brown.sents(),negative=5)
    # brnk = Word2VecKeras(brown.sents(),iter=3,negative=5)
    # print( brnk.most_similar('the', topn=5))
    # print( brn.most_similar('the', topn=5))

    # brcn = gensim.models.word2vec.Word2Vec(brown.sents(),sg=0,negative=5)
    # brcnk = Word2VecKeras(brown.sents(),iter=3,sg=0,negative=5)
    # print( brcnk.most_similar('the', topn=5))
    # print( brcn.most_similar('the', topn=5))
