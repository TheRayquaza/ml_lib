import numpy as np
import pytest
from linear.elasticnet import ElasticNet
from dataset.generic import generate_linear_dataset


@pytest.mark.parametrize("method", ["default", "stochastic", "mini-batch"])
def test_elasticnet_init(method):
    elasticnet = ElasticNet(method=method)
    assert elasticnet.method == method
    assert elasticnet.learning_rate == 1e-4
    assert elasticnet.decay == 1e-2
    assert elasticnet.alpha == 0.1
    assert elasticnet.rho == 1
    assert elasticnet.batch_size == 32
    assert elasticnet._fitted == False

    with pytest.raises(Exception, match="Unknown method"):
        ElasticNet(method="invalid_method")

    with pytest.raises(Exception, match="invalid batch size"):
        ElasticNet(method="mini-batch", batch_size=-1)


def test_elasticnet_str():
    elasticnet = ElasticNet()
    assert str(elasticnet) == "ElasticNet"


def test_elasticnet_compute_gradient():
    elasticnet = ElasticNet(method="default")
    elasticnet.X = np.random.rand(50, 3)
    elasticnet.y = np.random.rand(50, 1)
    elasticnet.weights = np.random.rand(3, 1)

    gradients = elasticnet._compute_gradient()

    assert gradients.shape == (3, 1)


def test_elasticnet_compute_stochastic_gradient():
    elasticnet = ElasticNet(method="stochastic")
    elasticnet.X = np.random.rand(50, 3)
    elasticnet.y = np.random.rand(50, 1)
    elasticnet.weights = np.random.rand(3, 1)

    gradients = elasticnet._compute_stochastic_gradient()

    assert gradients.shape == (3, 1)


def test_elasticnet_compute_mini_batch_gradient():
    elasticnet = ElasticNet(method="mini-batch")
    elasticnet.X = np.random.rand(50, 3)
    elasticnet.y = np.random.rand(50, 1)
    elasticnet.weights = np.random.rand(3, 1)

    gradients = elasticnet._compute_mini_batch_gradient()

    assert gradients.shape == (3, 1)


@pytest.mark.parametrize("method", ["default", "stochastic", "mini-batch"])
def test_elasticnet_train(method):
    X, y = generate_linear_dataset(200)
    elasticnet = ElasticNet(method=method)
    elasticnet.fit(X, y, epochs=10)

    assert elasticnet._fitted


def test_elasticnet_predict():
    X, y = generate_linear_dataset(200)
    elasticnet = ElasticNet()
    elasticnet.fit(X, y)

    predictions = elasticnet.predict(X)

    assert predictions.shape == y.shape
