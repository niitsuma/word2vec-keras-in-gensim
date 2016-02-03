#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import sys
import codecs

from os import path
import nltk
from nltk.tree import *

import numpy
import numpy as np
import copy
import random

import pickle

import csv


class TreeBank():
    def __init__(self,
                 dirpath='trees'
                 ):
        argdict= locals()
        argdict.pop('argdict',None)
        argdict.pop('self',None)
        vars(self).update(argdict)

    def load_tree_one(self,basename='dev'):
        trees=[]
        count=0
        infname = basename + '.txt'
        with codecs.open(path.join(self.dirpath, infname), 'r', 'utf-8') as f :
            for line in f.readlines():
                count = count +1
                #tree=Tree.parse(line)       ##nltk ver2
                tree =Tree.fromstring(line)  ##nltk ver3
                trees.append(tree)
        return trees


if __name__ == "__main__":
    treebank=TreeBank()
    t=treebank.load_tree_one()
    print t[:5]
