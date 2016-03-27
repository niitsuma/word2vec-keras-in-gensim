#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html

### ScoreWord2Vec stil has bug

import sys

import gensim

from word2veckeras.word2veckeras import Word2VecKeras
from word2veckeras.doc2veckeras import Doc2VecKeras,LabeledListSentence
from word2veckeras.scoreword2veckeras import ScoreWord2VecKeras,LineScoredWordSentence,ScoredListSentence

def dummy_score_vec_fn(word):
    return [len(word)/10.0,ord(word[0])/100.0,ord(word[-1])/100.0]
    #return [len(word)/0.2 ]


input_file = 'test.txt'
test_docs =gensim.models.doc2vec.TaggedLineDocument(input_file)
test_sents=gensim.models.word2vec.LineSentence(input_file)

dk0=Doc2VecKeras(test_docs,size=10,iter=3)
#sys.exit()

test_scorewordsents=LineScoredWordSentence(input_file,dummy_score_vec_fn)

### null_word must need for Doc2VecKeras(dm_concat=1)
vck = Word2VecKeras(test_sents,size=10,null_word=1,iter=3,sg=0)
#vck = Word2VecKeras(test_sents,null_word=0,iter=3,sg=0)
dklw=Doc2VecKeras(dm_concat=1)
print vck.syn0[0]
dklw.train_with_word2vec_instance(test_docs,vck,learn_words=True,iter=3)
#dk.train_with_word2vec_instance(test_docs,vck)
print dklw.syn0[0]
dk=Doc2VecKeras(dm_concat=1)
dk.train_with_word2vec_instance(test_docs,vck,learn_words=False,iter=3)
print dk.syn0[0]

#sys.exit()

svk=ScoreWord2VecKeras(test_scorewordsents,size=10,null_word=1,iter=3,sg=0)
print svk.syn0[0]
dsk=Doc2VecKeras(dm_concat=1)
dsk.train_with_word2vec_instance(test_docs,svk,learn_words=True,iter=3)
print dsk.syn0[0]

print(dk0.docvecs.most_similar(0))
print(dk.docvecs.most_similar(0))
print(dsk.docvecs.most_similar(0))
print(dklw.docvecs.most_similar(0))

#sys.exit()

from nltk.corpus import brown

brown_sents_sub=list(brown.sents()[:100])
brown_docs_sub=LabeledListSentence(brown_sents_sub)
brown_scorewordsents=list(ScoredListSentence(brown_sents_sub,dummy_score_vec_fn))


vck_br = Word2VecKeras(brown_sents_sub,null_word=1,iter=3,sg=0)
vkk_br = Word2VecKeras(brown_sents_sub,null_word=1,iter=3,sg=1)

dg_br=gensim.models.doc2vec.Doc2Vec(brown_docs_sub)
dk0_br=Doc2VecKeras(brown_docs_sub,iter=3)


svk_br=ScoreWord2VecKeras(brown_scorewordsents,null_word=1,iter=3,sg=0)

dk_br=Doc2VecKeras(dm_concat=1)
dk_br.train_with_word2vec_instance(brown_docs_sub,vck_br,learn_words=False,iter=3)

dkk_br=Doc2VecKeras(dm_concat=1)
dkk_br.train_with_word2vec_instance(brown_docs_sub,vkk_br,learn_words=False,iter=3,dm=1)

dsk_br=Doc2VecKeras(dm_concat=1)
dsk_br.train_with_word2vec_instance(brown_docs_sub,svk_br,learn_words=False,iter=3)


print(dg_br.docvecs.most_similar(0))
print(dk0_br.docvecs.most_similar(0))
print(dk_br.docvecs.most_similar(0))
print(dkk_br.docvecs.most_similar(0))
print(dsk_br.docvecs.most_similar(0))
