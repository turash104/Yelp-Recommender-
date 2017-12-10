from surprise import SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, Dataset, evaluate, Reader, GridSearch
import pandas as pd

# Loading Dataset
dftrain = pd.read_csv('train_rating_small.txt')
datatrain = Dataset.load_from_df(dftrain[['user_id', 'business_id', 'rating']], reader=Reader())
datatrain.split(n_folds=5)

# Training with Best Algo, Learning Parameters
#algo = SVD(n_epochs=15, n_factors=100, lr_all=0.008, reg_all=0.08)
#algo = SVDpp()
#algo = NMF()

#algo = SlopeOne()
algo = KNNWithZScore()
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
outfile = open("preds_small_5.csv", "w")
print("test_id,rating", file = outfile)
for k, v in predictions.items():
    print("{},{}".format(k, v), file=outfile)