from preprocessing.categorical_encoders import OneHotEncoder
import numpy as np

X = np.array([["a"] * 5 + ["b"] * 20 + ["c"] * 10 + ["d"] * 3], dtype=object).T
ohe = OneHotEncoder().fit(X)
print(ohe.infrequent_categories_)
print(ohe.transform(np.array([["a"]])))
print(ohe.transform(np.array([["b"]])))
print(ohe.transform(np.array([["c"]])))
print(ohe.transform(np.array([["d"]])))
