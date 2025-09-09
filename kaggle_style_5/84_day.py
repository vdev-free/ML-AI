import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report

# Дані
data = pd.DataFrame({
    "age": [22, 38, 26, 35, 54, 2, 27, 14, 30, 40],
    "income": [3000, 8000, 5000, 7000, 10000, 1500, 4000, 2000, 4500, 6500],
    "loan": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    "default": [0, 0, 1, 0, 1, 1, 0, 1, 0, 1]
})

X = data[["age", "income", "loan"]]
y = data["default"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Окремі моделі
log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Ensemble: Voting
voting = VotingClassifier(
    estimators=[("lr", log_reg), ("rf", rf)],
    voting="soft"  # "hard" = голосування, "soft" = ймовірності
)

voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)

print("Report:\n", classification_report(y_test, y_pred))
