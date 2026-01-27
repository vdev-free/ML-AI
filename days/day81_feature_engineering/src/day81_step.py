import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression



def make_data(n=10, seed=42):
    rng = np.random.default_rng(seed)

    country = rng.choice(["FI", "EE", "SE"], size=n)
    device = rng.choice(["mobile", "desktop"], size=n)
    plan = rng.choice(["free", "pro"], size=n)

    sessions = rng.poisson(lam=5, size=n)
    spent = rng.gamma(shape=2.0, scale=20.0, size=n)

    # логіка: хто більше користувався і має pro — частіше купує
    logit = (
        -3.0
        + 0.7 * (plan == "pro").astype(int)
        + 0.1 * sessions
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


def main():
    df = make_data(n=2000)

    # df["is_desktop"] = (df["device"] == "desktop").astype(int)
    # df["desktop_x_sessions"] = df["is_desktop"] * df["sessions"]

    df["is_pro"] = (df["plan"] == "pro").astype(int)
    df["pro_x_spent"] = df["is_pro"] * df["spent"]
    
    X = df.drop(columns=['y'])
    y = df["y"]

    cat_cols = ["country", "device", "plan"]
    num_cols = ["sessions", "spent", 'pro_x_spent']

    pre = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols),
        ]
    )

    model = LogisticRegression(max_iter=300)

    pipe = Pipeline(steps=[
        ("pre", pre),
        ('model', model),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')

    print("AUC scores:", [round(s, 4) for s in auc_scores])
    print("AUC mean:", round(float(auc_scores.mean()), 4))

if __name__ == "__main__":
    main()