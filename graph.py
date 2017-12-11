import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x =[1,2,3]
y1=[1.12879673951, 1.23056275714, 1.23351724944]
y2=[1.12586287617, 1.24678133000, 1.24427040532]
y3=[1.13289097159, 1.37991113222, 1.36945493389]
y4=[1.13289097159, 1.37615991251, 1.37592316229]
y5=[1.13289097159, 1.37185398912, 1.35482741696]
y6=[1.12773844155, 1.25010294701, 1.24314252807]
y7=[1.13289097159, 1.37561804478, 1.37401012058]
y8=[1.12619888539, 1.37353804480, 1.37070502194]
y9=[1.10649729974, 1.20822203355, 1.22014878243]

plt.plot(x, y1, label='SVD')
plt.plot(x, y2, label='SVDpp')
plt.plot(x, y3, label='NMF')
plt.plot(x, y4, label='SlopeOne')
plt.plot(x, y5, label='KNN Basic')
plt.plot(x, y6, label='KNN Baseline')
plt.plot(x, y7, label='KNN Means')
plt.plot(x, y8, label='KNN Z Score')
plt.plot(x, y9, label='Baseline ALS')
plt.ylabel('RMSE')
plt.legend()
plt.xlabel('Train Size')
plt.show()