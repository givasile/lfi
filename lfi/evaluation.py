import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from typing import Optional

def c2st(
        X: np.ndarray,
        Y: np.ndarray,
        seed: int = 1,
        n_folds: int = 5,
        scoring: str = "accuracy",
        z_score: bool = True,
        noise_scale: Optional[float] = None,
) -> float:
    '''Classifier-based 2-sample test returning accuracy
    
    Trains classifiers with N-fold cross-validation [1]. Scikit learn MLPClasifier are
    used, with 2 hidden layers of 10x dim each, where dim is the dimensionlaity of the
    samples X and Y.
    
    Code taken from sbi bm
    
    Args:
        X: Sample 1
        Y: Sample 2
        seed: Seed for sklearn
        n_folds: Number of folds
        z_score: Z-scoring using X
        noise_scale: If passed, will and Gaussian noise with std noise_scale to samples

    References:
        [1]: https://scikit-learn.org/stable/modules/cross_validation.html
    '''

    if z_score:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        X += noise_scale * np.random.randn(*X.shape)
        Y += noise_scale * np.random.randn(*Y.shape)

    ndim = X.shape[1]

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter = 10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((X, Y))
    target = np.concatenate(
        (
            np.zeros((X.shape[0],)),
            np.ones((Y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    score = np.mean(scores)
    return score

def c2st_auc(
        X: np.ndarray,
        Y: np.ndarray,
        seed: int = 1,
        n_folds: int = 5,
        z_score: bool = True,
        noise_scale: Optional[float] = None,
) -> float:
    '''Classifier-based 2 sample test returning AUC (area under curve)
    
    Same as c2st, except that it returns ROC AUC rather than accuracy
    
    Code taken from sbi bm
    Args: 
        X: Sample 1,
        Y: Sample 2,
        seed: Seed for sklearn,
        n_folds: Number of folds,
        z_score: Z-scoring using X,
        noise_scale: If passed, will add Gaussian noise with std noise_scale to samples
        
    Returns:
        Metric
    '''

    return c2st(
        X, 
        Y, 
        seed=seed,
        n_folds = n_folds,
        scoring = "roc_auc",
        z_score = z_score,
        noise_scale = noise_scale,
    )