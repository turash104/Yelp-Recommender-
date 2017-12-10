#from surprise import SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, Dataset, evaluate, Reader, GridSearch
import pandas as pd
import numpy as np

# Loading Dataset
dftrain = pd.read_csv('preds_ska.csv')
ratings = dftrain['rating']
round_ratings = np.round(ratings)
dftrain = dftrain.assign(rating = round_ratings)
print (dftrain)

dftrain.to_csv('preds_quantize.csv', index=False)
'''
# Printing Prediction Results
outfile = open("preds_quantize.csv", "w")
print("test_id,rating", file = outfile)
for k, v in predictions.items():
    print("{},{}".format(k, v), file=outfile)
'''