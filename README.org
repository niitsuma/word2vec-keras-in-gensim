* word2vec-keras-in-gensim

Just rewrite train function in gensim.models.word2vec.Word2Vec and gensim.models.doc2vec.Doc2Vec using Keras+Theano

like

#+BEGIN_SRC python
class Word2VecKeras(gensim.models.word2vec.Word2Vec):
     def train(...
#+END_SRC

And can use GPU via Theano. 

* Install
#+BEGIN_SRC bash
pip install word2veckeras
#+END_SRC

* Usage

same to gensim.models.word2vec.Word2Vec

** Example 
#+BEGIN_SRC python
vsk = Word2VecKeras(gensim.models.word2vec.LineSentence('test.txt'),iter=100)
print( vsk.most_similar('the', topn=5))

from nltk.corpus import brown
brk = Word2VecKeras(brown.sents(),iter=10)
print( brk.most_similar('the', topn=5))
#+END_SRC

* Requirements

#+BEGIN_SRC bash
pip install -U keras
#+END_SRC
