#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the GNU Affero General Public License, version 3 - http://www.gnu.org/licenses/agpl-3.0.html


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


def tree2label_sent(tree):
    ret=[]
    #x=[int(tree.node)  ,tree.leaves()] ##nltk ver2
    x=[int(tree.label()),tree.leaves()] ##nltk ver3
    ret.append(x)
    if len(tree)>1 :
        for t in tree:
            ret1=tree2label_sent(t)
            ret.extend(ret1)
    return ret

def label_sents2uni_sent(lss):
    lsis_join=[[ls[0],' '.join(ls[1]),i] for (i,ls) in enumerate(lss)]
    uss={}
    for lsi in lsis_join:
        if lsi[1] in uss:
            uss[lsi[1]].append(lsi[2])
        else:
            uss[lsi[1]]=[lsi[2]]
    return uss


def trees2label_sents(trees,only_root=False,pos_neg_label=False,remove_double_count_sentence=False):
    #print 'trees2label_sents',flag_word_lower,flag_stemmer,flag_remove_double_count_sentence,only_root,pos_neg_label
    #sys.exit()
    lss=[]
    for tree in trees:
        lss_tmp=tree2label_sent(tree)
        if pos_neg_label and lss_tmp[0][0] == 2 :
            continue
        if pos_neg_label :
            lss_tmp2 = [ [1 if ls[0] > 2 else 0  ,ls[1]] for ls in lss_tmp]
        else:
            lss_tmp2 =lss_tmp
        if len(lss_tmp2) > 0 and only_root:
            lss.append(lss_tmp2[0])
        elif len(lss_tmp2) > 0:
            lss.extend(lss_tmp2)
    if remove_double_count_sentence :
        uss=label_sents2uni_sent(lss)
        lss_new =[[np.mean([lss[id][0] for id in uss[s]]),lss[uss[s][0]][1] ] for s in uss ]
        return lss_new
    else:
        return lss


class TreeBank():
    def __init__(self,
                 dirpath='trees',
                 basenames=['train','test','dev'] 
                ):
        argdict= locals()
        argdict.pop('argdict',None)
        argdict.pop('self',None)
        vars(self).update(argdict)
        for basename in basenames:
            treename='tree_' + basename
            vars(self)[treename]=self.load_tree_one(basename)
        #print vars(self)

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
    def labeled_sents(self,basename='dev',only_root=False,pos_neg_label=False,remove_double_count_sentence=False):
        #treename='tree_' + basename
        return trees2label_sents(vars(self)['tree_' + basename],only_root=only_root,pos_neg_label=pos_neg_label,remove_double_count_sentence=remove_double_count_sentence)

    def sents_labels(self,basename='dev',only_root=0,pos_neg_label=0,remove_double_count_sentence=False):
        labeled_sents=self.labeled_sents(basename,only_root=only_root,pos_neg_label=pos_neg_label,remove_double_count_sentence=remove_double_count_sentence)
        X=[ls[1] for ls in labeled_sents]
        Y=[ls[0] for ls in labeled_sents]
        return X,Y


if __name__ == "__main__":
    treebank=TreeBank('./trees')
    print treebank.tree_dev[:3]
    # lss=trees2label_sents(treebank.tree_dev[:3])
    # print lss[:3]
    lss=treebank.labeled_sents('dev')[:3] ##fine grade
    print lss[:3]
    # lss=trees2label_sents(treebank.tree_dev[:5],only_root=1,pos_neg_label=1)
    # print lss[:3]


    # lss=trees2label_sents(treebank.tree_dev[:5],only_root=0,pos_neg_label=0)
    # print lss[:3]
    lss=treebank.labeled_sents('dev',only_root=True,pos_neg_label=True)[:3] ##fine grade
    print lss[:3]


