from surprise import SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, Dataset, evaluate, Reader, \
    GridSearch, Dataset, evaluate, Reader, GridSearch, BaselineOnly
import pandas as pd

# Loading Dataset
dftrain = pd.read_csv('train_rating.txt')
datatrain = Dataset.load_from_df(dftrain[['user_id', 'business_id', 'rating']], reader=Reader())
datatrain.split(n_folds=5)

# Tuning Hyper Parameters, Chosing Best Algo
#param_grid = {'n_epochs': [15], 'lr_all': [0.008, 0.01, 0.012, 0.014], 'reg_all': [0.1, 0.15, 0.2], 'n_factors': [1,2,3,4]}
#bsl_options = {'method': 'als', 'n_epochs': 15, 'reg_u': 6, 'reg_i': 2 }
param_grid = {'bsl_options': {'method': ['als'], 'n_epochs': [15,20], 'reg_u': [2,4,6,8], 'reg_i': [1,2,3] }}
grid_search = GridSearch(BaselineOnly, param_grid, measures=['RMSE'])
grid_search.evaluate(datatrain)
print('********************************')
print('Best Params: ' + str(grid_search.best_params['RMSE']))
print('Best Score: ' + str(grid_search.best_score['RMSE']))


# Training with Best Algo, Learning Parameters
algo = grid_search.best_estimator['RMSE']
#algo = BaselineOnly(bsl_options=bsl_options)
trainset = datatrain.build_full_trainset()
algo.train(trainset)

# Testing with Unknown Data, Making Prediction
dftest = pd.read_csv('test_rating.txt')
uids = dftest['user_id'].tolist()
iids = dftest['business_id'].tolist()
predictions = {}
for i, (uid, iid) in enumerate(zip(uids, iids)):
    prediction = algo.predict(uid=uid, iid=iid)
    predictions[i] = prediction.est

# Printing Prediction Results
outfile = open("baseLine_als_tuning_grid.csv", "w")
print("test_id,rating", file = outfile)
for k, v in predictions.items():
    print("{},{}".format(k, v), file=outfile)
