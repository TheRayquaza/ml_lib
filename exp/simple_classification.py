import numpy as np
import matplotlib.pyplot as plt
from dataset.classification import generate_classification_dataset
from graphic.curve import plot_2d_classification
from metrics import mse
from model_selection import train_test_split

models = []

X, y = generate_classification_dataset(n_samples=100, n_features=2, n_classes=2)

for model in models:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(X_train, y_train)
    print(model, ":")
    print("\tMSE on train:", mse(model.predict(X_train), y_train))
    print("\tMSE on test:", mse(model.predict(X_test), y_test))
    print("")

plot_2d_classification(X, y)
