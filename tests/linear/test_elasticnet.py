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


@pytest.mark.parametrize(
    "invalid_method, batch_size",
    [
        ("invalid_method", None),
        ("mini-batch", -1),
    ],
)
def test_elastic_fail_init(invalid_method, batch_size):
    if batch_size:
        with pytest.raises(ValueError):
            ElasticNet(method=invalid_method, batch_size=batch_size)
    else:
        with pytest.raises(ValueError):
            ElasticNet(method=invalid_method)

@pytest.mark.parametrize(
    "method, compute_gradient_function",
    [
        ("default", "_compute_gradient"),
        ("stochastic", "_compute_stochastic_gradient"),
        ("mini-batch", "_compute_mini_batch_gradient"),
    ],
)
def test_elasticnet_compute_gradients(method, compute_gradient_function):
    elasticnet = ElasticNet(method=method)
    elasticnet.X = np.random.rand(50, 3)
    elasticnet.y = np.random.rand(50, 1)
    elasticnet.weights = np.random.rand(3, 1)

    gradients = getattr(elasticnet, compute_gradient_function)()

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
