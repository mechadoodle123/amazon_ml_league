import lightgbm as lgb
import numpy as np

def create_model(n_estimators: int = 150, learning_rate: float = 0.08, max_depth: int = 6):
    """Instantiate a gradient boosted decision tree classifier."""
    return lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

def save_model(clf, filepath: str):
    """Save LightGBM booster model to text file."""
    clf.booster_.save_model(filepath)

def load_model(filepath: str):
    """Load LightGBM booster model from text file."""
    return lgb.Booster(model_file=filepath)

def predict_probabilities(booster, X: np.ndarray) -> np.ndarray:
    """Predict positive class match probability for a batch of feature vectors."""
    if len(X) == 0:
        return np.array([])
    # Booster.predict returns probability of positive class for binary objective
    return booster.predict(X)
