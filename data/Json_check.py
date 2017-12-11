import json

with open('final.json') as json_data:
    d = json.load(json_data)
    print(d)
