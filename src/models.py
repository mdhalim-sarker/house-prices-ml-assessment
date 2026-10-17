from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


def get_models():
    """
    Return the regression models used for experimentation.
    """

    models = {
        "Dummy Baseline": DummyRegressor(
            strategy="mean"
        ),

        "Ridge Regression": Ridge(
            alpha=10
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            random_state=42
        )
    }

    return models
