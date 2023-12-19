import numpy as np
from preprocessing.scalers import MinMaxScaler, StandardScaler, Normalizer
from sklearn.preprocessing import (
    MinMaxScaler as MinMaxScaler2,
    Normalizer as Normalizer2,
    StandardScaler as StandardScaler2,
)

# Sample data
data = np.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])
print("Normalized Data:")
print(Normalizer().fit_transform(data))
print(Normalizer2().fit_transform(data))
