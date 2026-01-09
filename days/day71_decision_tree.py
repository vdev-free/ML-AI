import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.tree import export_text

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# print('train', X_train.shape, y_train.shape)
# print('test', X_test.shape, y_test.shape)

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

y_pred = tree.predict(X_test)

# print("\n=== Decision Tree (no limits) ===")
# print("accuracy:", accuracy_score(y_test, y_pred))
# print("precision:", precision_score(y_test, y_pred))
# print("recall:", recall_score(y_test, y_pred))
# print("f1:", f1_score(y_test, y_pred))
# print("cm:\n", confusion_matrix(y_test, y_pred))

# print("\nTree depth:", tree.get_depth())
# print("Number of leaves:", tree.get_n_leaves())

# print("train accuracy:", tree.score(X_train, y_train))
# print("test accuracy:", tree.score(X_test, y_test))
# print(export_text(tree, feature_names=data.feature_names, max_depth=3))

# for d in [1, 2, 3, 4, 5, 7]:
#     tree_d = DecisionTreeClassifier(max_depth=d, random_state=42)
#     tree_d.fit(X_train, y_train)
    
#     train_acc = tree_d.score(X_train, y_train)
#     test_acc = tree_d.score(X_test, y_test)

#     print(f"\nmax_depth={d}")
#     print("train accuracy:", train_acc)
#     print("test accuracy:", test_acc)
#     print("depth:", tree_d.get_depth(), "leaves:", tree_d.get_n_leaves())

best_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
best_tree.fit(X_train, y_train)

y_pred = best_tree.predict(X_test)

print("\n=== Best Tree (max_depth=3) ===")
print("train accuracy:", best_tree.score(X_train, y_train))
print("test accuracy:", best_tree.score(X_test, y_test))
print("cm:\n", confusion_matrix(y_test, y_pred))
print("depth:", best_tree.get_depth(), "leaves:", best_tree.get_n_leaves())
