from clustering.gsdmm import MovieGroupProcess
import pandas as pd

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

docs = pd.read_json('data/train_review_10.json')
docs = docs.drop('id', 1)
docs = [text.split() for text in docs['text'].tolist()]
vocabulary_size = compute_V(texts=docs)
output = mgp.fit(docs, vocabulary_size)
print(output)



