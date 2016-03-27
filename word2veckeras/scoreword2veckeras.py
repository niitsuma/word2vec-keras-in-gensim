#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

import sys
import itertools
from Queue import Queue


from numpy import zeros, random, sum as np_sum, add as np_add, concatenate, \
    repeat as np_repeat, array, float32 as REAL, empty, ones, memmap as np_memmap, \
    sqrt, newaxis, ndarray, dot, vstack, dtype, divide as np_divide

import gensim.models.word2vec
import gensim.utils

from six.moves import xrange, zip
from six import string_types, integer_types, itervalues


import random

import numpy as np
import copy

import keras.constraints

from keras.utils.np_utils import accuracy
from keras.models import Graph,Sequential
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten , Lambda, Reshape,RepeatVector,Permute
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse


from word2veckeras import train_sg_pair, train_cbow_pair, queue_to_list , train_prepossess

def word_score2scored_word(word,score):
    return [word,score]
def scored_word2word(scored_word):
    return scored_word[0]
def scored_word2score(scored_word):
    return scored_word[1]


def train_batch_score_sg(model, scored_word_sentences,
                         score_vector_size,
                         alpha=None, work=None,
                         sub_batch_size=256,
                         batch_size=256):
    
    batch_count=0
    sub_batch_count=0
    train_x0 =np.zeros((batch_size,sub_batch_size),dtype='int32')
    train_x1 =np.zeros((batch_size,sub_batch_size),dtype='int32')
    train_y0  =np.zeros((batch_size,sub_batch_size),dtype='int8')
    train_y1  =np.zeros((batch_size,sub_batch_size,score_vector_size),dtype='float32')
    # train_x0=[[0]]*batch_size
    # train_x1=[[0]]*batch_size
    # train_y0=[[0]]*batch_size
    # train_y1=[[0]]*batch_size
    while 1:
        for scored_word_sentence in scored_word_sentences:
            #sentence=[scored_word2word(scored_word) for scored_word in scored_word_sentence]
            
            word_vocabs = [[model.vocab[w],s] for [w,s] in scored_word_sentence if w in model.vocab and
                           model.vocab[w].sample_int > model.random.rand() * 2**32]
            for pos, scored_word in enumerate(word_vocabs):
                reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
                word=scored_word2word(scored_word)
                # now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - model.window + reduced_window)
                for pos2, scored_word2 in enumerate(word_vocabs[start:(pos + model.window + 1 - reduced_window)], start):
                    word2=scored_word2word(scored_word2)
                    # don't train on the `word` itself
                    if pos2 != pos:
                        xy_gen=train_sg_pair(model, model.index2word[word.index], word2.index) #, alpha)
                        for xy in xy_gen :
                            if xy !=None:
                                (x0,x1,y0)=xy
                                y1=scored_word2score(scored_word)
                                train_x0[batch_count][sub_batch_count]=x0
                                train_x1[batch_count][sub_batch_count]=x1
                                train_y0[batch_count][sub_batch_count]=y0
                                train_y1[batch_count][sub_batch_count]=y1
                                sub_batch_count += 1
                                if sub_batch_count >= sub_batch_size :
                                    batch_count += 1
                                    sub_batch_count=0
                                if batch_count >= batch_size :
                                    yield { 'index':train_x0, 'point':train_x1, 'code':train_y0,'score':train_y1}
                                    batch_count=0

                                # train_x0[batch_count]=[x0]
                                # train_x1[batch_count]=x1
                                # train_y0[batch_count]=y0
                                # train_y1[batch_count]=y1
                                # #print train_x0,train_y1,
                                # batch_count += 1
                                # if batch_count >= batch_size :
                                #     #print { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y0),'score':np.array(train_y1)}
                                #     #yield { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y0),'score':np.array(train_y1,dtype=float32)}
                                #     yield { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y0),'score':np.array(train_y1)}
                                #     batch_count=0


def train_batch_score_cbow_xy_generator(model, scored_word_sentences):
    for scored_word_sentence in scored_word_sentences:
        #print scored_word_sentence
        scored_word_vocabs = [[model.vocab[w],s] for [w,s] in scored_word_sentence if w in model.vocab and  model.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, scored_word in enumerate(scored_word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(scored_word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [scored_word2[0].index for pos2, scored_word2 in window_pos if (scored_word2 is not None and scored_word2[0] is not None and pos2 != pos)]
            xy_gen=train_cbow_pair(model, scored_word[0] , word2_indices , None, None)
            for xy in xy_gen:
                if xy !=None:
                    xy1=[xy[0],xy[1],xy[2],[scored_word[1]]]
                    yield xy1

            # if xy !=None:
            #     xy1=[xy[0],xy[1],xy[2],scored_word[1]]
            #     yield xy1                           

def train_batch_score_cbow(model, scored_word_sentences, alpha=None, work=None, neu1=None,batch_size=100):
    w_len_queue_dict={}
    w_len_queue=[]

    while 1:
        for xy in train_batch_score_cbow_xy_generator(model, scored_word_sentences):
            if xy != None :
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
                    yield { 'index':np.array(train[0]),
                            'point':np.array(train[1]),
                            'code':np.array(train[2]),
                            'score':np.array(train[3])
                    }
                    w_len_queue=w_len_queue[1:]+[w_len_queue[0]]


                                
def build_keras_model_score_word_sg(index_size,vector_size,
                                    #vocab_size,
                                    context_size,
                                    #code_dim,
                                    score_vector_size,
                                    sub_batch_size=256,
                                    word_vectors=None,
                                    score_vectors=None,
                                    hidden_vectors=None,
                                    model=None
                                    ):
    """
    >>> word_vectors=np.array([[1,2,-1,1],[3,4,-1,-2],[5,6,-2,-2]])
    >>> score_vectors=np.array([[10,20,11,21,5,6,7,8],[30,40,33,41,9,8,7,6]])
    >>> hidden_vectors=np.array([[1,0,1,1],[0,1,1,1]])
    >>> sub_batch_size=3
    >>> vector_size=4
    >>> score_vector_size=2
    >>> kerasmodel=build_keras_model_score_word_sg(index_size=3,vector_size=vector_size,context_size=2,score_vector_size=score_vector_size,sub_batch_size=sub_batch_size,word_vectors=word_vectors,score_vectors=score_vectors,hidden_vectors=hidden_vectors)
    >>> ind=[[0,1,2],[1,2,0]]
    >>> ipt=[[1,0,1],[0,1,0]]
    >>> tmp1=kerasmodel.predict({'index':np.array(ind),'point':np.array(ipt)})
    >>> tmp3=np.array([[score_vectors[ipt[i][j]].reshape((score_vector_size,vector_size)).dot(word_vectors[ind[i][j]]) for j in range(sub_batch_size) ] for i in range(2)])
    >>> tmp2=np.array([[word_vectors[ind[i][j]].dot(hidden_vectors[ipt[i][j]].T) for j in range(sub_batch_size) ] for i in range(2)])
    >>> np.linalg.norm(1/(1+np.exp(-tmp2))-tmp1['code'])+np.linalg.norm(tmp1['score']-tmp3) < 0.0001
    True
    """
    
    kerasmodel = Graph()

    kerasmodel.add_input(name='point' , input_shape=(sub_batch_size,), dtype=int)
    kerasmodel.add_input(name='index' , input_shape=(sub_batch_size,), dtype=int)
    if word_vectors is None:
        kerasmodel.add_node(Embedding(index_size, vector_size, input_length=sub_batch_size                       ),name='embedding', input='index')
    else:
        kerasmodel.add_node(Embedding(index_size, vector_size, input_length=sub_batch_size,weights=[word_vectors]),name='embedding', input='index')
    if hidden_vectors is None:
        kerasmodel.add_node(Embedding(context_size, vector_size, input_length=sub_batch_size                        ),name='embedpoint', input='point')
    else:
        kerasmodel.add_node(Embedding(context_size, vector_size, input_length=sub_batch_size,weights=[hidden_vectors]),name='embedpoint', input='point')
    kerasmodel.add_node(Lambda(lambda x:x.sum(2))   , name='merge',inputs=['embedding','embedpoint'], merge_mode='mul')
    kerasmodel.add_node(Activation('sigmoid'), name='sigmoid', input='merge')
    kerasmodel.add_output(name='code',input='sigmoid')
    
    if score_vectors is None:
        kerasmodel.add_node(Embedding(context_size,  score_vector_size*vector_size, input_length=sub_batch_size,                       ),name='embedscore', input='point')
    else:
        kerasmodel.add_node(Embedding(context_size,  score_vector_size*vector_size, input_length=sub_batch_size,weights=[score_vectors]),name='embedscore', input='point')
    kerasmodel.add_node(Reshape((sub_batch_size,score_vector_size,vector_size,)) , name='score1',input='embedscore')
    
    kerasmodel.add_node(Flatten(), name='index1',input='embedding')
    kerasmodel.add_node(RepeatVector(score_vector_size), name='index2',input='index1')
    kerasmodel.add_node(Reshape((score_vector_size,sub_batch_size,vector_size,)) , name='index3',input='index2')
    kerasmodel.add_node(Permute((2,1,3,)) , name='index4',input='index3')
    
    kerasmodel.add_node(Lambda(lambda x:x.sum(-1))   , name='scorenode',inputs=['score1','index4'], merge_mode='mul')
    
    kerasmodel.add_output(name='score',input='scorenode')
    
    kerasmodel.compile('rmsprop', {'code':'mse','score':'mse'})
    return kerasmodel


def build_keras_model_score_word_cbow(index_size,vector_size,
                                      #vocab_size,
                                      context_size,
                                    #code_dim,
                                    score_vector_size,
                                    sub_batch_size=256,
                                    word_vectors=None,
                                    score_vectors=None,
                                    hidden_vectors=None,
                                    model=None,
                                    cbow_mean=False):

    """
    >>> word_vectors=np.array([[1,3,-1,2],[-2,4,-3,-1],[-3,4,2,-1]])
    >>> score_vectors=np.array([[10,20,11,21,5,6,7,8],[30,40,33,41,9,8,7,6]])
    >>> hidden_vectors=np.array([[-1,-1,1,-1],[1,-1,-1,1]])
    >>> sub_batch_size=3
    >>> vector_size=4
    >>> score_vector_size=2
    >>> kerasmodel=build_keras_model_score_word_cbow(index_size=3,vector_size=vector_size,context_size=2,score_vector_size=score_vector_size,sub_batch_size=sub_batch_size,word_vectors=word_vectors,score_vectors=score_vectors,hidden_vectors=hidden_vectors)
    >>> ind=[[0,1,0,2,1],[1,2,2,0,0]]
    >>> ipt=[[1,0,1],[0,1,0]]
    >>> tmp1=kerasmodel.predict({'index':np.array(ind),'point':np.array(ipt)})
    >>> tmp2=np.array([[word_vectors[ind[i]].sum(0).dot(hidden_vectors[ipt[i][j]].T) for j in range(sub_batch_size) ] for i in range(2)])
    >>> tmp3=np.array([[score_vectors[ipt[i][j]].reshape((score_vector_size,vector_size)).dot(word_vectors[ind[i]].sum(0)) for j in range(sub_batch_size) ] for i in range(2)])
    >>> np.linalg.norm(1/(1+np.exp(-tmp2))-tmp1['code'])+np.linalg.norm(tmp1['score']-tmp3) < 0.0001
    True
    """
    
    kerasmodel = Graph()
    kerasmodel.add_input(name='point' , input_shape=(sub_batch_size,), dtype=int)
    kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
    if word_vectors is None:
        kerasmodel.add_node(Embedding(index_size, vector_size,                       ),name='embedding', input='index')
    else:
        kerasmodel.add_node(Embedding(index_size, vector_size, weights=[word_vectors]),name='embedding', input='index')
    if hidden_vectors is None:
        kerasmodel.add_node(Embedding(context_size, vector_size, input_length=sub_batch_size                        ),name='embedpoint', input='point')
    else:
        kerasmodel.add_node(Embedding(context_size, vector_size, input_length=sub_batch_size,weights=[hidden_vectors]),name='embedpoint', input='point')

    if cbow_mean:
        kerasmodel.add_node(Lambda(lambda x:x.mean(1),output_shape=(vector_size,)),name='average',input='embedding')
    else:
        kerasmodel.add_node(Lambda(lambda x:x.sum(1) ,output_shape=(vector_size,)),name='average',input='embedding')
    
    kerasmodel.add_node(Activation('sigmoid'), name='sigmoid',inputs=['average','embedpoint'], merge_mode='dot',dot_axes=-1)
    kerasmodel.add_output(name='code',input='sigmoid')
    

    if score_vectors is None:
        kerasmodel.add_node(Embedding(context_size,  score_vector_size*vector_size, input_length=sub_batch_size,                       ),name='embedscore', input='point')
    else:
        kerasmodel.add_node(Embedding(context_size,  score_vector_size*vector_size, input_length=sub_batch_size,weights=[score_vectors]),name='embedscore', input='point')
    kerasmodel.add_node(Reshape((sub_batch_size,score_vector_size,vector_size,)) , name='score1',input='embedscore')
    #kerasmodel.add_node(Reshape((sub_batch_size,score_vector_size,vector_size,)) , name='scorenode',input='embedscore')

    
    ## kerasmodel.add_node(Flatten(), name='index1',input='average')
    kerasmodel.add_node(RepeatVector(score_vector_size*sub_batch_size), name='index2',input='average')
    kerasmodel.add_node(Reshape((score_vector_size,sub_batch_size,vector_size,)) , name='index3',input='index2')
    kerasmodel.add_node(Permute((2,1,3,)) , name='index4',input='index3')
    #kerasmodel.add_node(Permute((2,1,3,)) , name='scorenode',input='index3')
    
    kerasmodel.add_node(Lambda(lambda x:x.sum(-1))   , name='scorenode',inputs=['score1','index4'], merge_mode='mul')
    
    kerasmodel.add_output(name='score',input='scorenode')
    
    kerasmodel.compile('rmsprop', {'code':'mse','score':'mse'})
    return kerasmodel



class  ScoreWord2VecKeras(gensim.models.word2vec.Word2Vec):

    def scan_vocab(self, scored_word_sentences, progress_per=10000, trim_rule=None):
        scored_word_sentences1,        scored_word_sentences2        =itertools.tee(scored_word_sentences)
        
        sentences=(
            [
            #[scored_word2word(scored_word),scored_word2score(scored_word)]
            scored_word2word(scored_word)
            for scored_word in scored_word_sentence ]
                   for scored_word_sentence in scored_word_sentences1)
        super(ScoreWord2VecKeras, self).scan_vocab(sentences, progress_per, trim_rule)

        score_vec0=scored_word2score(scored_word_sentences2.next())
        self.score_vector_size=len(score_vec0[1])
        

    def train(self, scored_word_sentences,
              learn_doctags=True, learn_words=True, learn_hidden=True,iter=None,
              batch_size=128 #128, #512 #256
              ,sub_batch_size=128 #16 #32 #128 #128  #256 #128 #512 #256 #1
              #total_words=None, word_count=0,
              #chunksize=800,
              #total_examples=None, queue_factor=2, report_delay=1
              ):
        train_prepossess(self)
        vocab_size=len(self.vocab) 
        #batch_size=800 ##optimized 1G mem video card
        #batch_size=chunksize
        samples_per_epoch=int(self.window*2*sum(map(len,scored_word_sentences)))
        #print 'samples_per_epoch',samples_per_epoch
        if self.sg:
            #print 'sg',self.keras_context_index_size,sub_batch_size
            self.kerasmodel  =build_keras_model_score_word_sg(index_size=vocab_size,
                                                              vector_size=self.vector_size,
                                                              #vocab_size=vocab_size,
                                                              #code_dim=vocab_size,
                                                              context_size=self.keras_context_index_size,
                                                              score_vector_size=self.score_vector_size,
                                                              sub_batch_size=sub_batch_size,
                                                              model=self,
                                                              word_vectors=self.syn0,
                                                              hidden_vectors=self.keras_syn1,
                                                              )

            gen=train_batch_score_sg(self, scored_word_sentences, #self.alpha, work=None,
                                     score_vector_size=self.score_vector_size,
                                     sub_batch_size=sub_batch_size,
                                     batch_size=batch_size)
            self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)
        else:
            self.kerasmodel=build_keras_model_score_word_cbow(index_size=vocab_size,vector_size=self.vector_size,
                                                              # vocab_size=vocab_size,
                                                              # code_dim=vocab_size,
                                                              context_size=self.keras_context_index_size,
                                                              score_vector_size=self.score_vector_size,
                                                              sub_batch_size=1,#sub_batch_size,
                                                              model=self,
                                                              cbow_mean=self.cbow_mean,
                                                              word_vectors=self.syn0,
                                                              hidden_vectors=self.keras_syn1,
                                                              )

            #wv0=copy.copy(self.kerasmodel.nodes['embedding'].get_weights()[0][0])
            gen=train_batch_score_cbow(self, scored_word_sentences, self.alpha, work=None,batch_size=batch_size)
            self.kerasmodel.fit_generator(gen,samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)

        self.syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]
        if self.negative>0 and self.hs :
            syn1tmp=self.kerasmodel.nodes['embedpoint'].get_weights()[0]
            self.syn1=syn1tmp[0:len(self.vocab)]
            self.syn1neg=syn1tmp[len(self.vocab):2*len(self.vocab)]
        elif self.hs:
            self.syn1=self.kerasmodel.nodes['embedpoint'].get_weights()[0]
        else:
            self.syn1neg=self.kerasmodel.nodes['embedpoint'].get_weights()[0]






class LineScoredWordSentence(object):
    def __init__(self, source,score_fn, max_sentence_length=10000, limit=None):
        self.source = source
        self.score_fn=score_fn
        self.max_sentence_length = max_sentence_length
        self.limit = limit

    def __iter__(self):
        try:
            self.source.seek(0)
            for line in itertools.islice(self.source, self.limit):
                line = gensim.utils.to_unicode(line).split()
                i = 0
                while i < len(line):
                    yield [[w,self.score_fn(w)] for w in line[i : i + self.max_sentence_length]]
                    i += self.max_sentence_length
        except AttributeError:
            with gensim.utils.smart_open(self.source) as fin:
                for line in itertools.islice(fin, self.limit):
                    line = gensim.utils.to_unicode(line).split()
                    i = 0
                    while i < len(line):
                        yield  [[w,self.score_fn(w)] for w in line[i : i + self.max_sentence_length]]
                        i += self.max_sentence_length
       
class ScoredListSentence(object):
    def __init__(self, words_list,score_fn):
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
        self.score_fn=score_fn
    
    def __getitem__(self, index):
        t = [t for t in self]
        return t[index]
    def __iter__(self):
        for i, words in enumerate(self.words_list):
            #yield LabeledSentence(words, ['SENT_{0}'.format(i)])
            #yield gensim.models.doc2vec.TaggedDocument(words, [i])
            yield  [[w,self.score_fn(w)] for w in words]


                        

if __name__ == "__main__":
                                               
    import doctest
    doctest.testmod()
    
    input_file = 'test.txt'
    
    scales=[1.0,1.0,1.0]
    
    def dummy_score_vec(word):
        return [len(word)*scales[0],ord(word[0])*scales[1],ord(word[-1])*scales[1]]
        #return [len(word)/0.2 ]
        
    sws=list(LineScoredWordSentence(input_file,dummy_score_vec))
    #print sws[0]
    
    from word2veckeras import Word2VecKeras
    
    parameters = [{'size':[5],'hs':[0,1],'negative':[0,5],'sg':[0,1] }]
    from sklearn.grid_search import ParameterGrid
    for param in ParameterGrid(parameters):
        if (param['hs']==0 and param['negative']==0) :
            continue
        print param
        svk=ScoreWord2VecKeras(sws,**param)
        vsk = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),**param)
        vs = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file),**param)
        print( svk.most_similar('the', topn=5))
        print( vsk.most_similar('the', topn=5))
        print( vs.most_similar('the', topn=5))
        print(svk['the'])
        print(vsk['the'])
        print(vs['the'])

    # #svk.save_word2vec_format('tmp.vec')
    # #svk.save('tmp.model')

    #print svk.score_vector_size

    scored_word_list=[
        ['This',[20*0.1,10*0.2]],
        ['is',[10*0.1,5*0.2]],
        ['a',[30*0.1,10*0.2]],
        ['pen',[10*0.1,5*0.2]],
        ['.',[3*0.1,5*0.2]],
        ]
    
    scored_word_list=[scored_word_list]*100
    #print scored_word_list
    svk2=ScoreWord2VecKeras(scored_word_list,iter=3)
    print(svk2.most_similar('a',topn=5))
    #svk1.save('tmp.vec')
    #svk2.save_word2vec_format('tmp2.vec')

    
