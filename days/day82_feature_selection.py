import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

def make_data(n=5, seed=42):
    rng = np.random.default_rng(seed)
    sessions = rng.poisson(lam=5, size=n)
    spent = rng.gamma(shape=2.0, scale=20.0, size=n)
    noise1 = rng.normal(0, 1, size=n)
    noise2 = rng.normal(0, 1, size=n)

    logit = -3 + 0.3 * sessions + 0.01 * spent
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "sessions": sessions,
        "spent": spent,
        "noise1": noise1,
        "noise2": noise2,
        "y": y,
    })

    return df


df = make_data(n=2000)
# print(df)

X = df.drop(columns=["y"])
y = df["y"]

selector = SelectKBest(score_func=f_classif, k=2)
selector.fit(X, y)

mask = selector.get_support()
selected_features = X.columns[mask]

# print("Selected features (filter):", list(selected_features))

def auc_cv(X, y):
    model = LogisticRegression(max_iter=2000, penalty='l1', solver='liblinear')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    return float(scores.mean())

X_all = df[["sessions", "spent", "noise1", "noise2"]]
X_good = df[["sessions", "spent"]]
X_only_noise = df[["noise1", "noise2"]]
y = df["y"].to_numpy()

# print("Wrapper AUC (all):", round(auc_cv(X_all, y), 4))
# print("Wrapper AUC (good):", round(auc_cv(X_good, y), 4))
# print("Wrapper AUC (only noise):", round(auc_cv(X_only_noise, y), 4))

model_l1 = LogisticRegression(
    max_iter=4000,
    penalty="l1",
    solver="liblinear",
    C=0.03,
)

X_all = df[["sessions", "spent", "noise1", "noise2"]]
y = df["y"].to_numpy()

model_l1.fit(X_all, y)

coefs = pd.Series(model_l1.coef_[0], index=X_all.columns)
print("\nL1 coefficients:")
print(coefs.sort_values(key=lambda s: s.abs(), ascending=False))

selected = list(coefs[coefs != 0].index)
print("Embedded selected features:", selected)

