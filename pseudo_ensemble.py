from surprise import SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, Dataset, evaluate, Reader, \
    GridSearch, Dataset, evaluate, Reader, GridSearch
import pandas as pd

# Loading Dataset
dftrain = pd.read_csv('train_rating.txt')
datatrain = Dataset.load_from_df(dftrain[['user_id', 'business_id', 'rating']], reader=Reader())
datatrain.split(n_folds=5)
'''
param_grid = {'bsl_options': {'method': ['als', 'sgd'],
                              'reg': [1, 2]},
              'k': [2, 3],
              'n_epochs': [15],
              'lr_all': [0.008],
              'reg_all': [0.08],
              'sim_options': {'name': ['msd', 'cosine'],
                              'min_support': [1, 5],
                              'user_based': [False]}
              }
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])
grid_search.evaluate(datatrain)
print('********************************')
print('Best Params: ' + str(grid_search.best_params['RMSE']))
print('Best Score: ' + str(grid_search.best_score['RMSE']))
'''

# Tuning Hyper Parameters, Chosing Best Algo
param_grid = {'n_epochs': [15], 'lr_all': [0.008], 'reg_all': [0.08]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE'])
grid_search.evaluate(datatrain)
print('********************************')
print('Best Params: ' + str(grid_search.best_params['RMSE']))
print('Best Score: ' + str(grid_search.best_score['RMSE']))

# Tuning Hyper Parameters, Chosing Best Algo
param_grid = {'n_epochs': [15], 'lr_all': [0.008], 'reg_all': [0.08]}
grid_search = GridSearch(SVDpp, param_grid, measures=['RMSE'])
grid_search.evaluate(datatrain)
print('********************************')
print('Best Params: ' + str(grid_search.best_params['RMSE']))
print('Best Score: ' + str(grid_search.best_score['RMSE']))

# Tuning Hyper Parameters, Chosing Best Algo
param_grid = {'n_epochs': [15]}
grid_search = GridSearch(NMF, param_grid, measures=['RMSE'])
grid_search.evaluate(datatrain)
print('********************************')
print('Best Params: ' + str(grid_search.best_params['RMSE']))
print('Best Score: ' + str(grid_search.best_score['RMSE']))

# Tuning Hyper Parameters, Chosing Best Algo
param_grid = {}
grid_search = GridSearch(SlopeOne, param_grid, measures=['RMSE'])
grid_search.evaluate(datatrain)
print('********************************')
print('Best Params: ' + str(grid_search.best_params['RMSE']))
print('Best Score: ' + str(grid_search.best_score['RMSE']))

'''
# Training with Best Algo, Learning Parameters
algo = grid_search.best_estimator['RMSE']
trainset = datatrain.build_full_trainset()
algo.train(trainset)

# Testing with Unknown Data, Making Prediction
dftest = pd.read_csv('test_rating_small.txt')
uids = dftest['user_id'].tolist()
iids = dftest['business_id'].tolist()
predictions = {}
for i, (uid, iid) in enumerate(zip(uids, iids)):
    prediction = algo.predict(uid=uid, iid=iid)
    predictions[i] = prediction.est

# Printing Prediction Results
outfile = open("preds_small_10.csv", "w")
print("test_id,rating", file = outfile)
for k, v in predictions.items():
    print("{},{}".format(k, v), file=outfile)
'''
