#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

import sys
# is_py2 = sys.version[0] == '2'
# if is_py2:
#     import Queue as Queue
# else:
#     import queue as Queue
from Queue import Queue

from numpy import zeros, random, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide

import gensim.models.word2vec 

from six.moves import xrange, zip
from six import string_types, integer_types, itervalues


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


#global_count=0

def queue_to_list(q,extract_size):
    """ Dump a Queue to a list 
    >>> q=Queue()
    >>> q.put(1)
    >>> q.put(2)
    >>> q.put(3)
    >>> l=queue_to_list(q,2)
    >>> l==[1,2] and q.qsize() ==1
    True
    """
    # A new list
    l = []
    count=0
    while q.qsize() > 0:
        count +=1
        if count >extract_size:
            break
        l.append(q.get())

    return l



def train_sg_pair(model, word, context_index, alpha, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None,
                  scale=1.0
                  ):

    if word not in model.vocab:
        return
    predict_word = model.vocab[word]  # target word (NN output)
    
    if model.hs:
        y=np.zeros((len(model.vocab)), dtype=REAL)
        #y=np.zeros((len(model.vocab)), dtype='int8')
        x1code=np.zeros((len(model.vocab)), dtype='int8')
        x1code[predict_word.point]=1 #*scale
        #print np.linalg.norm(x1code-y),
        x1=predict_word.index
        y[predict_word.point]=predict_word.code
        x0=context_index
        #x1=predict_word.index
        return x0,x1,y

    if model.negative:
        x0=context_index
        y=np.zeros((len(model.vocab)), dtype='int8')
        x1=np.zeros((len(model.vocab)), dtype='int8')
        
        word_indices = [predict_word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != predict_word.index:
                word_indices.append(w)

        x1[word_indices]=1
        y[word_indices]=model.neg_labels
        return x0,x1,y ##missed in develop branch

def train_batch_sg(model, sentences, alpha, work=None,batch_size=100):
    
    batch_count=0

    train_x0=[[0]]*batch_size
    train_x1=[[0]]*batch_size
    train_y=[[0]]*batch_size

    while 1:
        #print( 'train_batch_sg')
        for i_sentence, sentence in enumerate(sentences):
            #print i_sentence,
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code

                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                window_length=len(word_vocabs[start:(pos + model.window + 1 - reduced_window)])
                #print window_length,
                for pos2, word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    # don't train on the `word` itself
                    if pos2 != pos:
                        xy=train_sg_pair(model, model.index2word[word.index], word2.index, alpha,scale=(window_length)/4)
                        if xy !=None:
                            (x0,x1,y)=xy
                            train_x0[batch_count]=[x0]
                            train_x1[batch_count]=[x1]
                            train_y[batch_count]=y
                            batch_count += 1
                            
                            if batch_count >= batch_size :
                                #yield { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y)}
                                yield [np.array(train_x0), np.array(train_x1)], np.array(train_y)
                                batch_count=0


def generate_model_point_embed_array(model,index_size,code_dim):
    vocab_size=len(model.vocab)
    # index_size=vocab_size
    # code_dim=vocab_size
    #model_point_embed=np.zeros((index_size,code_dim), dtype='int8')
    model_point_embed=np.zeros((index_size,code_dim), dtype=REAL)
    for i in range(index_size):
        #print model.index2word[i],model.vocab[model.index2word[i]].point
        model_point_embed[i][model.vocab[model.index2word[i]].point]=1.0
    # print model_point_embed
    # sys.exit()
    return model_point_embed

def build_keras_model_sg(index_size,vector_size,vocab_size,code_dim,learn_vectors=True,learn_hidden=True,model=None):
    model_point_embed=generate_model_point_embed_array(model,index_size,code_dim)

    
    word = Sequential()
    word.add(Embedding(index_size,vector_size, input_length=1,weights=[model.syn0]))
    word.add(Flatten())
    word.add(Dense(output_dim=code_dim,
                   b_constraint = keras.constraints.maxnorm(0),
                   weights=[model.syn1.T, np.zeros((code_dim),'float32')]
                   ))

    point = Sequential()
    point.add(Embedding(index_size,code_dim, input_length=1,trainable=False,
                        weights=[
                            #[[0,1,1],[0,1,1],[0,1,1]]
                            model_point_embed
                                 ]))
    point.add(Flatten())

    kerasmodel= Sequential()
    kerasmodel.add(Merge([word, point], mode='mul'))

    rmsprop1=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-04)
    kerasmodel.compile(loss='mse', optimizer=rmsprop1)
    #kerasmodel.compile(loss='mse', optimizer='rmsprop')
    #kerasmodel.compile(loss='mse', optimizer='sgd')
    #kerasmodel.compile(loss='binary_crossentropy', optimizer='sgd')
    
    # kerasmodel = Graph()
    # kerasmodel.add_input(name='point', input_shape=(code_dim,), dtype=REAL)
    # kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
    # kerasmodel.add_node(kerasmodel.inputs['point'],name='pointnode')
    # #kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1,weights=[model.syn0]),name='embedding', input='index')
    # kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1),name='embedding', input='index')
    # kerasmodel.add_node(Flatten(),name='embedflatten',input='embedding')
    # #kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0),weights=[model.syn1.T,np.zeros((code_dim))]), name='sigmoid', input='embedflatten')
    # kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0)), name='sigmoid', input='embedflatten')
    # kerasmodel.add_output(name='code',inputs=['sigmoid','pointnode'], merge_mode='mul')
    # kerasmodel.compile('rmsprop', {'code':'mse'})
    
    return kerasmodel



def train_cbow_pair(model, word, input_word_indices, l=None, alpha=None, learn_vectors=True, learn_hidden=True):
    
    if model.hs:
        x0=input_word_indices
        #x1=np.zeros((len(model.vocab)), dtype=REAL)
        #x1[word.point]=1
        x1=[word.index]
        y=np.zeros((len(model.vocab)), dtype='int8')
        y[word.point]=word.code
        return x0,x1,y
    
    if model.negative:
        word_indices = [word.index]
        while len(word_indices) < model.negative + 1:
            w = model.cum_table.searchsorted(model.random.randint(model.cum_table[-1]))
            if w != word.index:
                word_indices.append(w)
        x0=input_word_indices
        #x1=np.zeros((len(model.vocab)), dtype=REAL)
        
        x1[word_indices]=1
        y=np.zeros((len(model.vocab)), dtype=REAL)
        y[word_indices]=model.neg_labels
        return x0,x1,y
        


def train_batch_cbow_xy_generator(model, sentences):
    while 1:
        for sentence in sentences:
            #print sentence
            word_vocabs = [model.vocab[w] for w in sentence if w in model.vocab and  model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
                start = max(0, pos - model.window + reduced_window)
                window_pos = enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
                word2_indices = [word2.index for pos2, word2 in window_pos if (word2 is not None and pos2 != pos)]
                xy=train_cbow_pair(model, word , word2_indices , None, None)
                if xy !=None:
                    yield xy

def train_batch_cbow(model, sentences, alpha=None, work=None, neu1=None,batch_size=100):
    w_len_queue_dict={}
    w_len_queue=[]
    gen= train_batch_cbow_xy_generator(model, sentences)
    while 1:
        #print( 'train_batch_cbow')
        for xy in gen:
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
                    # yield { 'index':np.array(train[0]),
                    #         'point':np.array(train[1]),
                    #         'code':np.array(train[2])}
                    yield [np.array(train[0]),np.array(train[1])],np.array(train[2])

        
def build_keras_model_cbow(index_size,vector_size,vocab_size,code_dim,model=None,cbow_mean=False):
    
    model_point_embed=generate_model_point_embed_array(model,index_size,code_dim)
    #print model_point_embed
    word = Sequential()
    word.add(Embedding(index_size,vector_size,weights=[model.syn0]))
    word.add(Lambda(lambda x:x.sum(1),output_shape=(vector_size,)))
    word.add(Dense(output_dim=code_dim,
                   b_constraint = keras.constraints.maxnorm(0),
                   weights=[model.syn1.T, np.zeros((code_dim),'float32')]
                   ))
    point = Sequential()
    point.add(Embedding(index_size,code_dim, input_length=1,trainable=False,weights=[model_point_embed]))
    point.add(Flatten())

    kerasmodel= Sequential()
    kerasmodel.add(Merge([word, point], mode='mul'))

    kerasmodel.compile(loss='mse', optimizer='rmsprop')




    # kerasmodel = Graph()
    # kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
    # kerasmodel.add_input(name='point', input_shape=(code_dim,), dtype=REAL)
    # kerasmodel.add_node(kerasmodel.inputs['point'],name='pointnode')

    # #kerasmodel.add_node(Embedding(index_size, vector_size,weights=[model.syn0]),name='embedding', input='index')
    # #kerasmodel.add_node(Embedding(index_size, vector_size,trainable=True,weights=[model.syn0]),name='embedding', input='index')
    # kerasmodel.add_node(Embedding(index_size, vector_size),name='embedding', input='index')
    # #kerasmodel.add_node(Embedding(index_size, vector_size,trainable=True),name='embedding', input='index')


    
    # if cbow_mean:
    #     kerasmodel.add_node(Lambda(lambda x:x.mean(1),output_shape=(vector_size,)),name='average',input='embedding')
    # else:
    #     kerasmodel.add_node(Lambda(lambda x:x.sum(1),output_shape=(vector_size,)),name='average',input='embedding')
    # #kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0),weights=[model.syn1.T,np.zeros((code_dim))]), name='sigmoid', input='average')
    # #kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0),trainable=True,weights=[model.syn1.T,np.zeros((code_dim))]), name='sigmoid', input='average')
    # kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0)), name='sigmoid', input='average')
    # #kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0),trainable=True), name='sigmoid', input='average')
    # kerasmodel.add_output(name='code',inputs=['sigmoid','pointnode'], merge_mode='mul')
    # kerasmodel.compile('rmsprop', {'code':'mse'})

    # # print dir(kerasmodel.nodes['embedding'])
    # # #print kerasmodel.nodes['embedding']
    # # #print kerasmodel.nodes['embedding'].params
    # # #print kerasmodel.nodes['embedding'].updates
    # # print kerasmodel.nodes['embedding'].get_config()
    # # #print kerasmodel.nodes['embedding'].get_params()
    # # sys.exit()

    
    return kerasmodel

                            

class Word2VecKeras(gensim.models.word2vec.Word2Vec):

     def train(self, sentences, total_words=None, word_count=0, batch_size=800, total_examples=None, queue_factor=2, report_delay=1):
        vocab_size=len(self.vocab)
        #print 'Word2VecKerastrain'
        #batch_size=800 ##optimized 1G mem video card
        #batch_size=800
        batch_size=batch_size
        #batch_size=3200
        #samples_per_epoch=int(self.window*2*sum(map(len,sentences)))
        #print 'samples_per_epoch',samples_per_epoch

        if self.hs and self.negative>0 :
            raise ValueError("both using hs and negative not implemented") 
        
        if self.sg:
            if not hasattr(self, 'kerasmodel') :
                self.kerasmodel=build_keras_model_sg(index_size=vocab_size,vector_size=self.vector_size,vocab_size=vocab_size,code_dim=vocab_size,model=self)

            #print self.kerasmodel.predict([np.array([[0],[1]], "int32"),np.array([[0],[1]], "int32")])
            #sys.exit()
            
            samples_per_epoch=int(self.window*2*sum(map(len,sentences)))
            batch_size=100*self.window/2
            #print samples_per_epoch,batch_size

            # samples_per_epoch=5000
            # #samples_per_epoch=9000
            #batch_size=3

            gen= train_batch_sg(self, sentences, self.alpha, work=None,batch_size=batch_size)
            
            # wv0=copy.copy(self.kerasmodel.nodes['embedding'].get_weights()[0][0])
            #wv0=copy.copy(self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0])

            #self.kerasmodel.fit_generator(train_batch_sg(self, sentences, self.alpha, work=None,batch_size=batch_size),samples_per_epoch=samples_per_epoch, nb_epoch=self.iter, verbose=0)
            self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter, verbose=0)
            
            #self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)
            # count =0
            # for g in gen : #train_batch_sg(self, sentences, self.alpha, work=None,batch_size=batch_size):
            #     # print g[0]
            #     # print g[1]
            #     # print self.kerasmodel.predict(g[0])
            #     #wv0=copy.copy(self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0])
            #     self.kerasmodel.fit(g[0],g[1], nb_epoch=10, verbose=0)
            #     #print np.linalg.norm(wv0 - self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0]),
            #     #sys.exit()
            #     count +=1
            #     if count > self.iter * samples_per_epoch/batch_size :
            #         break

            
            # print wv0
            # # print self.kerasmodel.nodes['embedding'].get_weights()[0][0]
            # print self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0]
            # print np.linalg.norm(wv0 - self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0])
            # sys.exit()
            
            #self.syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]
            self.syn0=self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0]
        else:
            if not hasattr(self, 'kerasmodel'):
                self.kerasmodel=build_keras_model_cbow(index_size=vocab_size,vector_size=self.vector_size,vocab_size=vocab_size,code_dim=vocab_size,model=self,cbow_mean=self.cbow_mean)

            # print self.kerasmodel.predict([np.array([[0],[1]], "int32"),np.array([[0],[1]], "int32")])
            # sys.exit()
                
            samples_per_epoch=int(sum(map(len,sentences)))
            batch_size=100

            # samples_per_epoch=5
            # batch_size=3
            
            # #samples_per_epoch=5000
            # samples_per_epoch=9000
            # batch_size=60
            
            # debug_print_index=3
            # wv0=copy.copy(self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0])
            # wv00=copy.copy(self.kerasmodel.nodes['embedding'].get_weights()[0][debug_print_index])
            # wv0=copy.copy(self.kerasmodel.nodes['embedding'].get_weights()[0])
            # ws10=copy.copy(self.kerasmodel.nodes['sigmoid'].get_weights()[0][debug_print_index][debug_print_index+1])
            # #ws1=copy.copy(self.kerasmodel.nodes['sigmoid'].get_weights()[0][debug_print_index])
            # ws1=copy.copy(self.kerasmodel.nodes['sigmoid'].get_weights()[0])
            
            gen=train_batch_cbow(self, sentences, self.alpha, work=None,batch_size=batch_size)

            self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)

            # count =0
            # for g in gen : #train_batch_cbow(self, sentences, self.alpha, work=None,batch_size=batch_size):
            #     #print g[0]
            #     #print g[1]
            #     #print g['code'].shape,g['point'].shape,g['index']
            #     #print g['index']
            #     self.kerasmodel.fit(g[0],g[1], nb_epoch=1, verbose=0)
            #     #sys.exit()
            #     count +=1
            #     if count > self.iter * samples_per_epoch/batch_size :
            #         break

            # print wv00
            # print self.kerasmodel.nodes['embedding'].get_weights()[0][debug_print_index]
            # #print np.linalg.norm(wv0 - self.kerasmodel.nodes['embedding'].get_weights()[0][debug_print_index])
            # print np.linalg.norm(wv0 - self.kerasmodel.nodes['embedding'].get_weights()[0])
            # print ws10
            # print self.kerasmodel.nodes['sigmoid'].get_weights()[0][debug_print_index][debug_print_index+1]
            # #print self.kerasmodel.nodes['sigmoid'].get_weights()[0][debug_print_index]
            # #print np.linalg.norm(ws1 - self.kerasmodel.nodes['sigmoid'].get_weights()[0][debug_print_index])
            # print np.linalg.norm(ws1 - self.kerasmodel.nodes['sigmoid'].get_weights()[0])

            #print wv0
            # # print self.kerasmodel.nodes['embedding'].get_weights()[0][0]
            #print self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0]
            #print np.linalg.norm(wv0 - self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0])
            #sys.exit()
            
            
            #self.syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]
            self.syn0=self.kerasmodel.layers[0].layers[0].layers[0].get_weights()[0]
            


if __name__ == "__main__":

    import doctest
    doctest.testmod()
    global_count=0

    from nltk.corpus import brown #, movie_reviews, treebank

    input_file = 'test.txt'
    
    v_iter=1
    sg_v=1
    d_size=10
    ns=[1,10,20,50,100]
    
    sents=gensim.models.word2vec.LineSentence(input_file)
    sents=list(brown.sents())[0:2000]
    #sents=list(brown.sents())
    
    # vsk = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),iter=v_iter)

    #sys.exit()
    # vs = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file))

    # print( vs.most_similar('the', topn=5))
    
    # vck = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),sg=0,iter=v_iter)
    # vc = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file),sg=0)
    # print( vck.most_similar('the', topn=5))
    # print( vc.most_similar('the', topn=5))
    

    vs1 = gensim.models.word2vec.Word2Vec(sents,size=d_size,sg=sg_v,iter=1)
    #print( vs1['the'])
    print( vs1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))
    vs1 = gensim.models.word2vec.Word2Vec(sents,size=d_size,sg=sg_v,iter=10)
    #print( vs1['the'])
    print( vs1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))
    
    vsk1 = Word2VecKeras(sents,size=d_size,sg=sg_v,iter=1)
    #print( vsk1['the'])
    print( vsk1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))
    vsk1 = Word2VecKeras(sents,size=d_size,sg=sg_v,iter=5)
    #print( vsk1['the'])
    print( vsk1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))
    #vsk1 = Word2VecKeras(sents,size=d_size,sg=sg_v,iter=v_iter)
    ##print( vsk1['the'])
    #print( vsk1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))
    vsk1.train(sents)
    #print( vsk1['the'])
    print( vsk1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))
    vsk1.train(sents)
    #print( vsk1['the'])
    print( vsk1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))

    for n in ns :
        print( n)
        vsk.iter=n
        vsk.train(brown_sents)
        print( vsk1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))

    
    # vck1 = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),sg=0,size=5,iter=v_iter)
    # vc1 = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file),sg=0,size=5,iter=3)
    # print vck1['the']
    # print vc1['the']

    sys.exit()

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
    #brown_sents=brown.sents()
    #brown_sents=list(brown.sents()[:10000])
    brown_sents=list(brown.sents())[:2000]
    v_iter=1

    # brck = Word2VecKeras(brown_sents,iter=2,sg=0,size=4,window=3)
    # sys.exit()
    
    # br = gensim.models.word2vec.Word2Vec(brown_sents)
    # print br.most_similar_cosmul(positive=['woman', 'he'], negative=['man'], topn=10)
    # brk = Word2VecKeras(brown_sents,iter=v_iter)
    # #print( brk.most_similar('the', topn=5))
    # #print( br.most_similar('the', topn=5))
    # #print brk.most_similar_cosmul(positive=['france', 'england'], negative=['london'], topn=10)
    # #print br.most_similar_cosmul(positive=['france', 'england'], negative=['london'], topn=10)
    # #print brk.most_similar(positive=['woman', 'king'], negative=['man'])
    # #print br.most_similar(positive=['woman', 'king'], negative=['man'])
    # #print br.most_similar_cosmul(positive=['woman', 'husband'], negative=['man'], topn=10)
    # #print brk.most_similar_cosmul(positive=['woman', 'husband'], negative=['man'], topn=10)
    # print brk.most_similar_cosmul(positive=['woman', 'he'], negative=['man'], topn=10)
    
    

    # for x, y in model.most_similar(positive=["香川"], negative=["うどん"], topn=3):
    #     print x, y
    #sys.exit()
    ns=[1,10,20,50,100]
    #ns=[2]
    # ns=[200,400,1000]
    # for n in ns :
    #     print n
    #     brc = gensim.models.word2vec.Word2Vec(brown_sents,sg=0,iter=n)
    #     print brc.most_similar_cosmul(positive=['woman', 'he'], negative=['man'], topn=4)
    # sys.exit()
    brk1 = Word2VecKeras(brown_sents,size=40,iter=1,sg=sg_v)
    #print brk.most_similar_cosmul(positive=['woman', 'he'], negative=['man'], topn=4)
    print( brk1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))
    for n in ns :
        print( n)
        brk1.iter=n
        brk1.train(brown_sents)
        #print brk.most_similar_cosmul(positive=['woman', 'he'], negative=['man'], topn=4)
        print( brk1.most_similar_cosmul(positive=['she', 'him'], negative=['he'], topn=4))
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
