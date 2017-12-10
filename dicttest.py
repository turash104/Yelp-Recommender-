import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from random import randint
import ast

'''
dftrain = pd.read_csv('out')
dftrain = dftrain.groupby('business_id')
f = {'train_id':[lambda x: list(x)], 'text':[lambda x: '. '.join(x)]}
dftrain = dftrain.agg(f)
'''
#df = pd.read_json('first_ten_train_reviews.json')
s = open('first_ten_train_reviews.json', 'r').read()
self.whip = ast.literal_eval(s)




