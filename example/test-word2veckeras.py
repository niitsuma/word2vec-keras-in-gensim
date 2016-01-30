import gensim
from word2veckeras.word2veckeras import Word2VecKeras

input_file = 'test.txt'

v_iter=1
v_size=5
sg_v=1
topn=4

vs1 = gensim.models.word2vec.Word2Vec(gensim.models.word2vec.LineSentence(input_file),sg=sg_v,size=v_size,iter=1)
print( vs1.most_similar('the', topn=topn))
print vs1['the']
vsk1 = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),sg=sg_v,size=v_size,iter=1)
print vsk1['the']
print( vsk1.most_similar('the', topn=topn))
vsk1 = Word2VecKeras(gensim.models.word2vec.LineSentence(input_file),sg=sg_v,size=v_size,iter=5)
print vsk1['the']
print( vsk1.most_similar('the', topn=topn))

