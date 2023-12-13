import matplotlib.pyplot as plt

def display_curve(X, y):
    fig, ax = plt.subplots()
    ax.scatter(X, y)
    ax.set(xlim=(min(X), max(X)), ylim=(min(y), max(y)))
    plt.show()
