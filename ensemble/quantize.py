#from surprise import SVD, SVDpp, NMF, SlopeOne, KNNBasic, KNNWithMeans, KNNBaseline, KNNWithZScore, Dataset, evaluate, Reader, GridSearch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading Dataset
answer1 = pd.read_csv('preds.csv')
answer2 = pd.read_csv('SVDpp.csv')
answer3 = pd.read_csv('baseLine.csv')
id = answer1['test_id']
ratings1 = answer1['rating']
ratings2 = answer2['rating']
ratings3 = answer3['rating']
avg_ratings = (np.array(ratings1) + np.array(ratings2) + np.array(ratings3)) / 3
n=50
plt.plot(id[:n], ratings1[:n], label='SVD SGD')
plt.plot(id[:n], ratings2[:n], label='SVDpp SGD')
plt.plot(id[:n], ratings3[:n], label='Baseline Only ALS')
#plt.plot(id[:n], avg_ratings[:n], label='ensemble')
plt.ylabel('RMSE')
plt.legend()
plt.xlabel('Test id')
plt.show()

'''
# Printing Prediction Results
outfile = open("preds_ensemble.csv", "w")
print("test_id,rating", file = outfile)
for k, v in zip(id, avg_ratings):
    print("{},{}".format(k, v), file=outfile)
'''