import numpy as np
import math
import random
import pandas as pd
import datetime
import collections

def getlowhigh(data,index):
    keys=data.keys()
    feature = sorted(data[keys[index]])
    low= feature[0]
    high = feature[len(feature) - 1]
    return (low,high)

def pearson(dict1,dict2):
    keys = list()
    for key in dict1.keys():
        if(key in keys):
            continue
        else:
            keys.append(key)

    for key in dict2.keys():
        if(key in keys):
            continue
        else:
            keys.append(key)
    return np.corrcoef(
        [dict1.get(x, 0) for x in keys],
        [dict2.get(x, 0) for x in keys])[0, 1]


data = pd.read_csv('train_rating.txt')

keys=data.keys()
data = data[data[keys[4]]>'2016-12-30']

numsplits=10
splitPoints=list()
for i in range(1,numsplits):
    splitPoints.append(int(i*len(data)/numsplits))
splits=np.split(data, splitPoints)

threshold=0
sum_err=0.0
for i in range(0, numsplits):
    test = splits[i]
    test = test.reset_index(drop=True)

    before = splits[:i]
    after = splits[i + 1:]
    train = pd.concat(before + after)
    train = train.reset_index(drop=True)

    user_profile = collections.defaultdict(dict)
    item_profile = collections.defaultdict(dict)

    for index, row in train.iterrows():
        user_index = row[keys[1]]
        item_index = row[keys[2]]
        rating = row[keys[3]]
        user_profile[user_index][item_index] = rating
        item_profile[item_index][user_index] = rating

    err=0.0
    for index, row in test.iterrows():
        user_index = row[keys[1]]
        item_index = row[keys[2]]
        true_rating = row[keys[3]]

        candidates=item_profile[item_index]
        peers=list()
        sumw=0
        for candidate in candidates:
            sim=math.fabs(pearson(user_profile[user_index], user_profile[candidate]))
            if(sim>threshold):
                sumw+=sim
                peers.append((sim,candidates[candidate]))

        estimated_rating=0.0
        for peer in peers:
            estimated_rating+=float(peer[0]*peer[1])/float(sumw)

        err+=(true_rating-estimated_rating)*(true_rating-estimated_rating)

    sum_err+=err

avg_err=sum_err/numsplits
print(avg_err)