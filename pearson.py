import random
from surprise import accuracy
from surprise import SVD
from surprise import dataset
from surprise import Dataset
from surprise import evaluate, print_perf, Reader, GridSearch
import pandas as pd


reader = Reader(rating_scale=(1, 5), sep=',')

dftrain = pd.read_csv('train_rating.txt')
dftrain.drop('train_id', 1)
dftrain.drop('date', 1)
data = Dataset.load_from_df(dftrain[['user_id', 'business_id', 'rating']], reader)
raw_ratings = data.raw_ratings
random.shuffle(raw_ratings)
data.raw_ratings = raw_ratings  # data is now the set A
data.split(n_folds=3)

print('Grid Search...')
param_grid = {'n_epochs': [10], 'lr_all': [0.008], 'reg_all': [0.2]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'], verbose=0 )
grid_search.evaluate(data)
algo = grid_search.best_estimator['RMSE']
trainset = data.build_full_trainset()
algo.train(trainset)

dftest = pd.read_csv('test_rating.txt')
dftest.drop('test_id', 1)
dftest.drop('date', 1)
dftest['rating'] = 0
datatest = Dataset.load_from_df(dftest[['user_id', 'business_id', 'rating']], reader)

predictions = algo.test(trainset.build_testset())
print(accuracy.rmse(predictions))


# Compute unbiased accuracy on B
testset = data.construct_testset(B_raw_ratings)  # testset is now the set B
predictions = algo.test(testset)
print(accuracy.rmse(predictions))