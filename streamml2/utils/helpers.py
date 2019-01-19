# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 09:24:39 2019

@author: bmccs
"""
from collections import defaultdict

def listofdict2dictoflist(somelistofdicts):
    nd=defaultdict(list)
    for d in somelistofdicts:
       for key,val in d.items():
          nd[key].append(val)
    return nd