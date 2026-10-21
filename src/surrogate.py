
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel, RBF, WhiteKernel
)


def train_surrogate(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()

    if len(X_train) < 5 or len(X_train) != len(y_train):
        raise ValueError("Insufficient or inconsistent training data.")

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(
            length_scale=np.ones(X_train.shape[1]),
            length_scale_bounds=(1e-2, 1e2),
        )
        + WhiteKernel(
            noise_level=1e-5,
            noise_level_bounds=(1e-8, 1e-1),
        )
    )

    model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)
    return model
