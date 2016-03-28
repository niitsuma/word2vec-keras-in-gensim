import sys
import numpy as np
import gensim
from word2veckeras.scoreword2veckeras import ScoreWord2VecKeras,LineScoredWordSentence,ScoredListSentence


def compare_w2v(w2v1,w2v2):
    s=0.0
    count =0
    for w in w2v1.vocab:
        if w in w2v2.vocab:
            d=np.linalg.norm(w2v1[w]-w2v2[w])
            count +=1
            s += d
    return s/count

input_file = 'test.txt'
   
scales=[1.0,1.0,1.0]
def dummy_score_vec(word):
    return [len(word)*scales[0],ord(word[0])*scales[1],ord(word[-1])*scales[1]]
    #return [len(word)/0.2 ]

v_iter=1
v_size=5
sg_v=1
topn=4

sws=list(LineScoredWordSentence(input_file,dummy_score_vec))
svk=ScoreWord2VecKeras(sws,hs=1,negative=0,sg=sg_v,size=v_size,iter=1)
vs = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file),hs=1,negative=0,sg=sg_v,size=v_size,iter=1)

print( svk.most_similar('the', topn=5))
print( vs.most_similar('the', topn=5))
print(svk['the'])
print(vs['the'])

#svk.save_word2vec_format('tmp.vec')
#svk.save('tmp.model')

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



from nltk.corpus import brown
brown_sents=list(brown.sents())[:200]
#brown_sents=list(brown.sents())

vs = gensim.models.word2vec.Word2Vec(brown_sents)
svk2=ScoreWord2VecKeras(ScoredListSentence(brown_sents,dummy_score_vec))


print( vs.most_similar('the', topn=5))
print( svk.most_similar('the', topn=5))
