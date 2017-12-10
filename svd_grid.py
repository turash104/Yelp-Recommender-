from surprise import SVD, Dataset, evaluate, Reader, GridSearch
import pandas as pd
import numpy as np
from collections import defaultdict
from random import randint


type_of_business = defaultdict(list)

# Loading Dataset
dftrain = pd.read_csv('train_rating.txt')
dftrain = dftrain[dftrain['date']>'2016-12-30']
dftrain = dftrain.reset_index()
dftrain_size = dftrain.shape[0]

type=[]
for index, row in dftrain.iterrows():
    u_id, b_id, r = row['user_id'], row['business_id'], row['rating']
    t = randint(0, 14)
    type.append(t)
    type_of_business[b_id].append(t)

dftrain = dftrain.assign(type=type)

dftrains = []
for i in range(15):
    temp = dftrain[dftrain['type'] == i]
    temp = temp.reset_index()
    temp = temp.drop('level_0', 1)
    temp = temp.drop('index', 1)
    dftrains.append(temp)

algos = []
for i in range(15):
    df = dftrains[i]
    datatrain = Dataset.load_from_df(df[['user_id', 'business_id', 'rating']], reader=Reader())
    datatrain.split(n_folds=10)

    # Tuning Hyper Parameters, Chosing Best Algo
    #param_grid = {'n_epochs': [30, 40, 50], 'lr_all': [0.002, 0.004, 0.06, 0.008], 'reg_all': [0.02, 0.04, 0.06, 0.08]}
    param_grid = {'n_epochs': [5], 'lr_all': [0.008], 'reg_all': [0.08]}
    grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])
    grid_search.evaluate(datatrain)
    print('********************************')
    print('Best Params: ' + str(grid_search.best_params['RMSE']))
    print('Best Score: ' + str(grid_search.best_score['RMSE']))

    # Training with Best Algo, Learning Parameters
    algo = grid_search.best_estimator['RMSE']
    trainset = datatrain.build_full_trainset()
    algo.train(trainset)
    algos.append(algo)

# Testing with Unknown Data, Making Prediction
dftest = pd.read_csv('test_rating.txt')
uids = dftest['user_id'].tolist()
iids = dftest['business_id'].tolist()
predictions = {}
for i, (uid, iid) in enumerate(zip(uids, iids)):
    cnt = np.bincount(type_of_business[iid])
    if len(cnt)==0:
        print (i,uid, iid)
        continue
    iid_type = cnt.argmax()
    algo = algos[iid_type]
    prediction = algo.predict(uid=uid, iid=iid)
    predictions[i] = prediction.est

# Printing Prediction Results
outfile = open("test.csv", "w")
print("test_id,rating", file = outfile)
for k, v in predictions.items():
    print("{},{}".format(k, v), file=outfile)
