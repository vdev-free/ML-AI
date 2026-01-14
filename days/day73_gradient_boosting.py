import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

def report(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("accuracy:", accuracy_score(y_true, y_pred))
    print("precision:", precision_score(y_true, y_pred))
    print("recall:", recall_score(y_true, y_pred))
    print("f1:", f1_score(y_true, y_pred))
    print("cm:\n", confusion_matrix(y_true, y_pred))

# 1) Decision Tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
report("Decision Tree (max_depth=3)", y_test, tree.predict(X_test))

# 2) Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
report("Random Forest (200 trees)", y_test, rf.predict(X_test))
print("RF train acc:", rf.score(X_train, y_train), "RF test acc:", rf.score(X_test, y_test))

# 3) Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42)
gb.fit(X_train, y_train)
report("Gradient Boosting (200, lr=0.05, depth=2)", y_test, gb.predict(X_test))
print("GB train acc:", gb.score(X_train, y_train), "GB test acc:", gb.score(X_test, y_test))

for lr in [0.2, 0.1, 0.05, 0.02]:
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=lr,
        max_depth=2,
        random_state=42
    )
    gb.fit(X_train, y_train)
    y_pred = gb.predict(X_test)

    print(f"\n=== GB lr={lr} ===")
    print("train acc:", gb.score(X_train, y_train))
    print("test acc:", gb.score(X_test, y_test))
    print("cm:\n", confusion_matrix(y_test, y_pred))

gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=2, random_state=42)
gb.fit(X_train, y_train)
report("Gradient Boosting (300, lr=0.05, depth=2)", y_test, gb.predict(X_test))
print("GB train acc:", gb.score(X_train, y_train), "GB test acc:", gb.score(X_test, y_test))

gb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=2, random_state=42)
gb.fit(X_train, y_train)
report("Gradient Boosting (400, lr=0.05, depth=2)", y_test, gb.predict(X_test))
print("GB train acc:", gb.score(X_train, y_train), "GB test acc:", gb.score(X_test, y_test))

gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=2, random_state=42)
gb.fit(X_train, y_train)
report("Gradient Boosting (500, lr=0.05, depth=2)", y_test, gb.predict(X_test))
print("GB train acc:", gb.score(X_train, y_train), "GB test acc:", gb.score(X_test, y_test))
