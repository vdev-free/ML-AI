import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def make_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)

    country = rng.choice(["FI", "EE", "SE"], size=n, p=[0.5, 0.3, 0.2])
    device = rng.choice(["mobile", "desktop"], size=n, p=[0.7, 0.3])
    plan = rng.choice(["free", "pro"], size=n, p=[0.7, 0.3])

    sessions = rng.poisson(lam=5, size=n)
    spent = rng.gamma(shape=2.0, scale=20.0, size=n)

    # таргет: купив/не купив (просто логіка)
    logit = (
        -3.0
        + 0.7 * (plan == "pro").astype(int)
        + 0.2 * (device == "desktop").astype(int)
        + 0.12 * sessions
        + 0.01 * spent
    )
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame(
        {
            "country": country,
            "device": device,
            "plan": plan,
            "sessions": sessions,
            "spent": spent,
            "y": y,
        }
    )
    return df


def eval_auc(df: pd.DataFrame) -> float:
    X = df.drop(columns=["y"])
    y = df["y"].to_numpy()

    cat_cols = ["country", "device", "plan"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    pipe = Pipeline(
        [
            ("pre", pre),
            ("model", LogisticRegression(max_iter=300)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return float(cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean())


def main():
    df = make_data()

    # 1) чесно
    auc_clean = eval_auc(df)
    print("AUC clean:", round(auc_clean, 4))

    # 2) додаємо leakage (підглядання відповіді)
    rng = np.random.default_rng(0)
    df_leak = df.copy()
    df_leak["leaky_feature"] = df_leak["y"] + rng.normal(0, 0.05, size=len(df_leak))

    auc_leak = eval_auc(df_leak)
    print("AUC with leakage:", round(auc_leak, 4))


if __name__ == "__main__":
    main()
