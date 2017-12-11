from clustering.gsdmm import MovieGroupProcess
import pandas as pd
from collections import defaultdict


def compute_V(texts):
    V = set()
    # print(texts)
    for sentences in texts:
        for word in sentences:
            # print(word)
            # print("====SET===")
            V.add(word)
    # print(V)
    # print(texts)
    return len(V)

# Lower alpha more people per group
mgp = MovieGroupProcess(K=15, alpha=0.000001, beta=1.1, n_iters=200)

docs = pd.read_json('data/final.json')
ratings = pd.read_csv('train_rating.txt')


docs['business_id'] = ratings['business_id']
docs


def f(x):
     return pd.Series(dict(id = list(x['id']),
                        text = " ".join(x['text'])
                        ))

docs = docs.groupby('business_id').agg(f)


# f = {'id': [lambda x: list(x)], 'text': [lambda x: '. '.join(x)]}
# docs = docs.agg(f)
#
# docs
# docs['text']
#
# for key in docs.keys():
#     print(key)
#
reviews  = [text.split() for text in docs['text'].tolist()]
vocabulary_size = compute_V(texts=docs)
output = mgp.fit(docs, vocabulary_size)
# print(output)

output_Vector = pd.Series(output)
docs["cluster_id"] = output_Vector

# docs

cluster_dict = {}

for cluster_id, t_ids in zip(docs['cluster_id'],docs['id']):
    print(cluster_id,t_ids)
    if cluster_id not in cluster_dict:
        cluster_dict[cluster_id] = t_ids
    else:
        for x in t_ids:
            cluster_dict[cluster_id].append(x)

print( cluster_dict)
