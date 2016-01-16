#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

import gensim

from word2veckeras.word2veckeras import Word2VecKeras
from word2veckeras.doc2veckeras import Doc2VecKeras,LabeledListSentence
from word2veckeras.scoreword2veckeras import ScoreWord2VecKeras,LineScoredWordSentence

input_file = 'test.txt'
test_docs =gensim.models.doc2vec.TaggedLineDocument(input_file)
test_sents=gensim.models.word2vec.LineSentence(input_file)
    
def dummy_score_vec(word):
    return [len(word)/10.0,ord(word[0])/100.0]
    #return [len(word)/0.2 ]

test_scorewordsents=LineScoredWordSentence(input_file,dummy_score_vec)


### null_word must need for concat 
vck = Word2VecKeras(test_sents,null_word=1,iter=3,sg=0)
dk0=Doc2VecKeras(test_docs,iter=3)
dk=Doc2VecKeras(dm_concat=1)
#print vck.syn0[0]
dk.train_with_word2vec_instance(test_docs,vck,learn_words=False,iter=3)
#dk.train_with_word2vec_instance(test_docs,vck)
#print dk.syn0[0]

svk=ScoreWord2VecKeras(test_scorewordsents,null_word=1,iter=3,sg=0)
dsk=Doc2VecKeras(dm_concat=1)
dsk.train_with_word2vec_instance(test_docs,svk,learn_words=False,iter=3)
#print dsk.syn0[0]

print(dk0.docvecs.most_similar(0))
print(dk.docvecs.most_similar(0))
print(dsk.docvecs.most_similar(0))


