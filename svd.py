import optparse, sys, os, logging
from surprise import SVD
from surprise import dataset
from surprise import Dataset
from surprise import evaluate, print_perf,Reader
import pandas as pd
import csv


# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.

df = pd.read_csv('train_rating.txt')
reader = Reader(rating_scale=(0, 5),sep=',')
df.drop('train_id',1)
df.drop('date',1)
data = Dataset.load_from_df(df[['user_id', 'business_id', 'rating']], reader)

trainset=data.build_full_trainset()



# We'll use the famous SVD algorithm.
algo = SVD(n_factors=101,biased=True)
algo.train(trainset)
testset=trainset.build_testset()
predictions=algo.test(testset)


with open('predicted_rating.csv', 'w') as csvfile:
    fieldnames = ['test_id', 'rating']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, lineterminator = '\n')

    writer.writeheader()


    for idx, row in enumerate(predictions):
        print(row[3])
        writer.writerow({'test_id': idx, 'rating': row[3]})








