import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

def score_feature(
    feature: pd.Series,
    y: pd.Series,
    model,
    min_cells: int = 20,
    test_size: float = 0.33
) -> float:
    """
    Score one feature against y using a fitted model.

    Only uses cells where feature > 0 and at least `min_cells` cells.
    Splits data into train/test, fits model, and returns test R^2 score.
    """
    nonzero = feature > 0
    if nonzero.sum() < min_cells:
        return 0.0

    X_sub = feature[nonzero].to_frame()
    y_sub = y[nonzero]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub, test_size=test_size
    )
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


def compute_pt_corr(
    X: pd.DataFrame,
    y: pd.Series,
    model=None,
    min_cells: int = 20,
    test_size: float = 0.33,
    cv: int = 3,
    n_jobs: int = 1
) -> pd.Series:
    """
    Compute non-linear correlation scores for each column in X against y.

    Parameters:
    - X: DataFrame (samples × features).
    - y: Series (samples).
    - model: scikit-learn estimator or GridSearchCV. If None, uses default spline+ridge.
    - min_cells: minimum non-zero observations to include a feature.
    - test_size: fraction for test split.
    - cv: number of folds for hyperparameter search (if using default).
    - n_jobs: parallel jobs for GridSearchCV.

    Returns:
    - Series of R² scores indexed by feature name.
    """
    if model is None:
        pipe = Pipeline([
            ("sp", SplineTransformer(degree=3)),
            ("rr", Ridge())
        ])
        param_grid = {
            "sp__n_knots": [2, 3],
            "rr__alpha": [1e-3, 1e-1, 10]
        }
        model = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=n_jobs)

    # Apply the scoring function to each column
    scores = X.apply(
        score_feature,
        y=y,
        model=model,
        min_cells=min_cells,
        test_size=test_size
    )
    return scores
