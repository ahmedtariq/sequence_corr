# sequence_corr
A lightweight Python package to measure non-linear relationships between features (e.g., gene expression) and a continuous variable (e.g., pseudotime).

## Usage

```python
import pandas as pd
from sequence_corr import compute_pt_corr

# X: DataFrame of shape (n_samples, n_features) e.g: (cells, genes)
# y: Series of sequence values (same index as X)
# Optionally pass your own model (GridSearchCV pipeline)
scores = compute_pt_corr(
    X, y,
    model=None,           # uses a default spline+ridge pipeline with hyperparameter search
    min_cells=20,         # minimum non-zero observations per feature
    test_size=0.33,
    cv=3,
    n_jobs=1
)

# scores is a Series indexed by features e.g: genes, containing model.score values
``` 

## API Reference

See `sequence_corr/core.py` for docs on each function.
