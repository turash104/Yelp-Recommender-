import sys
import pandas as pd
import numpy as np
from collections import defaultdict

train_data = pd.read_csv('train_rating.txt')
test_data = pd.read_csv('test_rating.txt')

user_profile = defaultdict(float)
user_profile_count = defaultdict(float)
business_profile = defaultdict(float)
business_profile_count = defaultdict(float)

avg = np.average(train_data['rating'])

for index, row in train_data.iterrows():
    if index%10000 == 0:
        print(index/10000)
    u_id, b_id, r = row['user_id'], row['business_id'], row['rating']
    user_profile[u_id] = user_profile.get(u_id, 0.0) + r - avg
    user_profile_count[u_id] = user_profile_count.get(u_id, 0.0) + 1
    business_profile[b_id] = business_profile.get(b_id, 0.0) + r -avg
    business_profile_count[b_id] = business_profile_count.get(b_id, 0.0) + 1


print(avg)
user_keys = user_profile.keys()
for key in user_keys:
    user_profile[key] = user_profile[key] / user_profile_count[key]

business_keys = business_profile.keys()
print(len(business_profile))
for key in business_keys:
    business_profile[key] = business_profile[key] / business_profile_count[key]

rating_list=[]
for index, row in test_data.iterrows():
    if index%10000 == 0:
        print(index/10000)
    val=avg + user_profile.get(row['user_id'], 0.0) + business_profile.get(row['business_id'], 0.0)
    if val > 5:
        val = 5
    elif val < 1:
        val = 1
    rating_list.append(val)
test_data = test_data.assign(rating = rating_list )

output = test_data['rating']
output.to_csv('nothing.csv')