import numpy

dict1 = {'car': 0.1, 'dog':0.3, 'tiger':0.5, 'lion': 0.1, 'fish':0.2}
dict2 = {'goat':0.3, 'fish':0.3, 'shark':0.4, 'dog':0.3}

keys = list(dict1.keys() | dict2.keys())
print(numpy.corrcoef(
    [dict1.get(x, 0) for x in keys],
    [dict2.get(x, 0) for x in keys])[0, 1])