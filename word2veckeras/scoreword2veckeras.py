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
from keras.layers.core import Dense, Dropout, Activation, Merge, Flatten , Lambda
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.objectives import mse


from word2veckeras import train_sg_pair, train_cbow_pair, queue_to_list

def word_score2scored_word(word,score):
    return [word,score]
def scored_word2word(scored_word):
    return scored_word[0]
def scored_word2score(scored_word):
    return scored_word[1]


def train_batch_score_sg(model, scored_word_sentences, alpha=None, work=None,batch_size=100):
    
    batch_count=0

    train_x0=[[0]]*batch_size
    train_x1=[[0]]*batch_size
    train_y0=[[0]]*batch_size
    train_y1=[[0]]*batch_size

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
                        xy=train_sg_pair(model, model.index2word[word.index], word2.index, alpha)
                        if xy !=None:
                            (x0,x1,y0)=xy
                            y1=scored_word2score(scored_word)
                            train_x0[batch_count]=[x0]
                            train_x1[batch_count]=x1
                            train_y0[batch_count]=y0
                            train_y1[batch_count]=y1
                            #print train_x0,train_y1,
                            batch_count += 1
                            if batch_count >= batch_size :
                                #print { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y0),'score':np.array(train_y1)}
                                #yield { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y0),'score':np.array(train_y1,dtype=float32)}
                                yield { 'index':np.array(train_x0), 'point':np.array(train_x1), 'code':np.array(train_y0),'score':np.array(train_y1)}
                                batch_count=0


def train_batch_score_cbow_xy_generator(model, scored_word_sentences):
    for scored_word_sentence in scored_word_sentences:
        #print scored_word_sentence
        scored_word_vocabs = [[model.vocab[w],s] for [w,s] in scored_word_sentence if w in model.vocab and  model.vocab[w].sample_int > model.random.rand() * 2**32]
        for pos, scored_word in enumerate(scored_word_vocabs):
            reduced_window = model.random.randint(model.window)  # `b` in the original word2vec code
            start = max(0, pos - model.window + reduced_window)
            window_pos = enumerate(scored_word_vocabs[start:(pos + model.window + 1 - reduced_window)], start)
            word2_indices = [scored_word2[0].index for pos2, scored_word2 in window_pos if (scored_word2 is not None and scored_word2[0] is not None and pos2 != pos)]
            xy=train_cbow_pair(model, scored_word[0] , word2_indices , None, None)
            if xy !=None:
                xy1=[xy[0],xy[1],xy[2],scored_word[1]]
                yield xy1                           

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


                                
def build_keras_model_score_word_sg(index_size,vector_size,vocab_size,code_dim,score_vector_size,sub_batch_size=256,model=None):

    kerasmodel = Graph()

    kerasmodel.add_input(name='point' , input_shape=(sub_batch_size,), dtype=int)
    kerasmodel.add_input(name='index' , input_shape=(sub_batch_size,), dtype=int)
    kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1,weights=[model.syn0]),name='embedding', input='index')
    kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1,weights=[model.syn1]),name='embedpoint', input='point')

    

    kerasmodel.add_input(name='point', input_shape=(code_dim,), dtype=REAL)
    kerasmodel.add_node(kerasmodel.inputs['point'],name='pointnode')
    
    kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)

    kerasmodel.add_node(Embedding(index_size, vector_size, input_length=1),name='embedding', input='index')
    kerasmodel.add_node(Flatten(),name='embedflatten',input='embedding')
    kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0)), name='sigmoid', input='embedflatten')
    kerasmodel.add_output(name='code',inputs=['sigmoid','pointnode'], merge_mode='mul')

    kerasmodel.add_node(Dense(score_vector_size,b_constraint = keras.constraints.maxnorm(0)), name='scorenode', input='embedflatten')
    kerasmodel.add_output(name='score',input='scorenode')
    kerasmodel.compile('rmsprop', {'code':'mse','score':'mse'})
    return kerasmodel



def build_keras_model_score_word_cbow(index_size,vector_size,vocab_size,code_dim,score_vector_size,model=None,cbow_mean=False):

    kerasmodel = Graph()

    kerasmodel.add_input(name='point', input_shape=(code_dim,), dtype=REAL)
    kerasmodel.add_node(kerasmodel.inputs['point'],name='pointnode')
    
    kerasmodel.add_input(name='index' , input_shape=(1,), dtype=int)
    
    kerasmodel.add_node(Embedding(index_size, vector_size),name='embedding', input='index')
    if cbow_mean:
        kerasmodel.add_node(Lambda(lambda x:x.mean(1),output_shape=(vector_size,)),name='average',input='embedding')
    else:
        kerasmodel.add_node(Lambda(lambda x:x.sum(1),output_shape=(vector_size,)),name='average',input='embedding')
    kerasmodel.add_node(Dense(code_dim, activation='sigmoid',b_constraint = keras.constraints.maxnorm(0)), name='sigmoid', input='average')
    kerasmodel.add_output(name='code',inputs=['sigmoid','pointnode'], merge_mode='mul')

    kerasmodel.add_node(Dense(score_vector_size,b_constraint = keras.constraints.maxnorm(0)), name='scorenode', input='average')
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
        

    def train(self, scored_word_sentences, total_words=None, word_count=0, chunksize=800, total_examples=None, queue_factor=2, report_delay=1):
        vocab_size=len(self.vocab) 
        #print 'ScoreWord2VecKeras.train'

        #batch_size=800 ##optimized 1G mem video card
        batch_size=chunksize
        samples_per_epoch=int(self.window*2*sum(map(len,scored_word_sentences)))
        #print 'samples_per_epoch',samples_per_epoch
        if self.sg:
            self.kerasmodel  =build_keras_model_score_word_sg(index_size=vocab_size,vector_size=self.vector_size,vocab_size=vocab_size,code_dim=vocab_size,score_vector_size=self.score_vector_size,model=self)

            #wv0=copy.copy(self.kerasmodel.nodes['embedding'].get_weights()[0][0])
            
            self.kerasmodel.fit_generator(train_batch_score_sg(self, scored_word_sentences, self.alpha, work=None,batch_size=batch_size),samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)

            # print wv0
            # print self.kerasmodel.nodes['embedding'].get_weights()[0][0]
            # sys.exit()
            
            self.syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]
        else:
            self.kerasmodel=build_keras_model_score_word_cbow(index_size=vocab_size,vector_size=self.vector_size,vocab_size=vocab_size,code_dim=vocab_size,
                                                              score_vector_size=self.score_vector_size,
                                                              model=self,
                                                              cbow_mean=self.cbow_mean
                                                              )

            #wv0=copy.copy(self.kerasmodel.nodes['embedding'].get_weights()[0][0])
            
            self.kerasmodel.fit_generator(train_batch_score_cbow(self, scored_word_sentences, self.alpha, work=None,batch_size=batch_size),samples_per_epoch=samples_per_epoch, nb_epoch=self.iter,verbose=0)

            # print wv0
            # print self.kerasmodel.nodes['embedding'].get_weights()[0][0]
            # sys.exit()

            self.syn0=self.kerasmodel.nodes['embedding'].get_weights()[0]






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

    # model=build_keras_model_score_word_cbow(index_size=3,vector_size=4,vocab_size=3,code_dim=5,score_vector_size=2,model=None)
    # sys.exit()
    
    input_file = 'test.txt'

    n_iter=3
    size=10
    scales=[1.0,1.0,1.0]
    
    def dummy_score_vec(word):
        return [len(word)*scales[0],ord(word[0])*scales[1],ord(word[-1])*scales[1]]
        #return [len(word)/0.2 ]
        
    sws=list(LineScoredWordSentence(input_file,dummy_score_vec))
    # print sws[0]
    
    #svk=ScoreWord2VecKeras(sws)
    svk=ScoreWord2VecKeras( LineScoredWordSentence(input_file,dummy_score_vec),size=size,iter=n_iter,sg=0)
    
    #svk.save_word2vec_format('tmp.vec')
    #svk.save('tmp.model')

    from word2veckeras import Word2VecKeras
    vsk = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),size=size,iter=n_iter)
    #vs = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file))

    print( svk.most_similar('the', topn=5))
    print( vsk.most_similar('the', topn=5))
    #print( vs.most_similar('the', topn=5))
    print(svk['the'])
    print(vsk['the'])
    
    sys.exit()
        
    #svck = ScoreWord2VecKeras(LineScoredWordSentence(input_file,dummy_score_vec),size=3,iter=1,sg=0)
    svck  = ScoreWord2VecKeras(LineScoredWordSentence(input_file,dummy_score_vec),iter=1,sg=0)
    print( svck.most_similar('the', topn=5))
    #sys.exit()
    

    


    
    #print svk.score_vector_size
    #model1=build_keras_model_score_word_sg(index_size=3,vector_size=2,vocab_size=2,code_dim=2,score_vector_size=2)

    #print( svk.most_similar('the', topn=8))

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

    # from ScoreSent2Vec.word2vec import ScoredSent2Vec,Sent2Vec,LineSentence
    # #print list(LineSentence(input_file))
    # sv1=Sent2Vec(LineSentence(input_file),model_file='tmp.model')
    
