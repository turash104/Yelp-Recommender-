from surprise import SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, Dataset, evaluate, Reader, \
    GridSearch, Dataset, evaluate, Reader, GridSearch, BaselineOnly
import pandas as pd

# Loading Dataset
dftrain = pd.read_csv('train_rating.txt')
datatrain = Dataset.load_from_df(dftrain[['user_id', 'business_id', 'rating']], reader=Reader())
datatrain.split(n_folds=5)

# Tuning Hyper Parameters, Chosing Best Algo
# param_grid = {'n_epochs': [15], 'lr_all': [0.01], 'reg_all': [0.15], 'n_factors': [2]}
bsl_options = {'method': 'als','n_epochs': 15, 'reg_u': 6, 'reg_i' : 2 }

algo = BaselineOnly(bsl_options=bsl_options)
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
outfile = open("baseLine.csv", "w")
print("test_id,rating", file = outfile)
for k, v in predictions.items():
    print("{},{}".format(k, v), file=outfile)
