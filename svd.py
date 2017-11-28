import optparse, sys, os, logging
from surprise import SVD
from surprise import dataset
from surprise import Dataset
from surprise import evaluate, print_perf, Reader, GridSearch
import pandas as pd
import csv

# Loading the trainset
dftrain = pd.read_csv('train_rating.txt')
# Loading the testset
dftest = pd.read_csv('test_rating.txt')

# The Reader class is used to parse a file containing ratings - one rating per line with a fixed structure
reader = Reader(rating_scale=(1, 5), sep=',')
#Pre-processing trainset to drop 2 columns
dftrain.drop('train_id', 1)
dftrain.drop('date', 1)
# Pre-processing testset to drop 2 columns
dftest.drop('test_id', 1)
dftest.drop('date', 1)
# Pre-processing testset to add a dummy column containing rating = 0 as the last column
dftest['rating'] = 0

# dftrain = dftrain.sample(frac=1)

# Creating a Dataset object for both train and test sets
datatrain = Dataset.load_from_df(dftrain[['user_id', 'business_id', 'rating']], reader)
datatest = Dataset.load_from_df(dftest[['user_id', 'business_id', 'rating']], reader)
datatrain.split(n_folds=5)

# param_grid = {'n_epochs': [5, 10, 15], 'lr_all': [0.002, 0.005, 0.008],
#               'reg_all': [0.2, 0.4, 0.6, 0.8]}
# grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])

param_grid = {'n_epochs': [15], 'lr_all': [0.005],
              'reg_all': [0.2]}

# TODO Try other error measures
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])

# best RMSE score
print(grid_search.best_score['RMSE'])
# >>> 0.96117566386

# combination of parameters that gave the best RMSE score
print(grid_search.best_params['RMSE'])
# >>> {'reg_all': 0.4, 'lr_all': 0.005, 'n_epochs': 10}

# Building the trainset
trainset = datatrain.build_full_trainset()
# Building the testset like the we built the trainset -- prolly can do away with this step:
testset = datatest.build_full_trainset()
# Building the testset
testset = testset.build_testset()
grid_search.evaluate(datatrain)

# creating the algo object: In this case it's an SVD type. This is where the various parameters need to be passed
# algo = grid_search.best_estimator['RMSE']
algo = SVD()
# Training the algorithm using the full train set
algo.train(trainset)
# Making the predictions
predictions = algo.predict(testset)
predictions = pd.DataFrame(predictions)
# Printing the predictions into a file with submission format

predictions.to_csv("predictions.csv")



# with open('predicted_rating_grid.csv', 'w') as csvfile:
#     fieldnames = ['test_id', 'rating']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')
#     writer.writeheader()
#     for idx, row in enumerate(predictions):
#         print(row[3])
#         writer.writerow({'test_id': idx, 'rating': row[3]})
