import numpy as np

from sklearn.metrics import mean_squared_error


def log_rmse(y_true, y_pred):
    """
    Calculate RMSE for values already transformed
    using log1p(SalePrice).
    """

    return float(
        np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    )


def absolute_errors(y_true, y_pred):
    """
    Calculate absolute errors between actual and
    predicted values on the same scale.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return np.abs(y_true - y_pred)
