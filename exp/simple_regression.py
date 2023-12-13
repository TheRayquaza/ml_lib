import numpy as np
import matplotlib.pyplot as plt
from regression.elasticnet import ElasticNet
from regression.lasso import Lasso
from regression.linear import LinearModel
from regression.ridge import Ridge
from preprocessing import PolynomialFeatures, StandardScaler, Normalizer
from dataset.generic import generate_linear_dataset, generate_polynomial_dataset
from graphic import display_curve
from metrics import mse
from model_selection import train_test_split

models = [
    LinearModel(),
    Lasso(),
    Ridge(),
    ElasticNet()
]

X, y = generate_linear_dataset(300)
display_curve(X, y)

for model in models:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    print(model, ":")
    print("\tMSE on train:", mse(model.predict(X_train), y_train))
    print("\tMSE on test:", mse(model.predict(X_test), y_test))
    print("")
