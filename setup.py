from setuptools import setup
from setuptools import find_packages

setup(name='word2veckeras',
      version='0.0.4.2',
      description='word2vec based on Kearas and gensim',
      author='Hirotaka Niitsuma',
      author_email='hirotaka.niitsuma@gmail.com',
      url='https://github.com/niitsuma/word2vec-keras-in-gensim',
      download_url='https://github.com/niitsuma/word2vec-keras-in-gensim/archive/master.zip',
      license='GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html',
      #install_requires=['six'],
      install_requires=['gensim', 'theano', 'pyyaml', 'six', 'keras'],
      #install_requires=['gensim', 'theano', 'pyyaml', 'six', 'keras', 'sklearn'],
      #install_requires=['gensim', 'theano',  'keras'],
      # extras_require={
      #     'h5py': ['h5py'],
      # },
      packages=find_packages(),
      test_suite = 'test'
      )
