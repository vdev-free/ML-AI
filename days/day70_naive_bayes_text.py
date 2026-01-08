from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer


texts = [
    "Win money now",                 # spam
    "Free prize claim",              # spam
    "Limited offer win win",         # spam
    "Call now to get free bonus",    # spam
    "Meeting tomorrow at 10",        # ham
    "Project deadline is next week", # ham
    "Invoice attached please review",# ham
    "Let us schedule a call",        # ham
]

y = [1, 1, 1, 1, 0, 0, 0, 0]  # 1=spam, 0=not spam (ham)

X_train, X_test, y_train, y_test = train_test_split(
    texts, y, test_size=0.25, random_state=42, stratify=y
)

# print("train size:", len(X_train))
# print("test size:", len(X_test))
# print("train labels:", y_train)
# print("test labels:", y_test)

model = Pipeline(steps=[
    ('vect', CountVectorizer()),
    ('nb', MultinomialNB())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# print("\nTEST TEXTS:", X_test)
# print("Pred:", y_pred)
# print("True:", y_test)

# print("\naccuracy:", accuracy_score(y_test, y_pred))
# print("precision:", precision_score(y_test, y_pred, zero_division=0))
# print("recall:", recall_score(y_test, y_pred, zero_division=0))
# print("f1:", f1_score(y_test, y_pred, zero_division=0))
# print("cm:\n", confusion_matrix(y_test, y_pred))

# Дістаємо частини з pipeline
vect = model.named_steps["vect"]
nb = model.named_steps["nb"]

# print("\nVocabulary size:", len(vect.vocabulary_))

# Ймовірності для тесту: [P(ham), P(spam)]
proba = model.predict_proba(X_test)
# print("Probabilities [P(ham), P(spam)] for each test text:\n", proba)

# Подивимось топ-слівця для spam/ham
feature_names = vect.get_feature_names_out()

spam_log_prob = nb.feature_log_prob_[1]  # слова для класу 1 (spam)
ham_log_prob = nb.feature_log_prob_[0]   # слова для класу 0 (ham)

top_spam_idx = spam_log_prob.argsort()[-8:][::-1]
top_ham_idx = ham_log_prob.argsort()[-8:][::-1]

# print("\nTop words for SPAM:")
# for i in top_spam_idx:
#     print(feature_names[i])

# print("\nTop words for HAM:")
# for i in top_ham_idx:
#     print(feature_names[i])

# print("\nX_test:")
# for t, p in zip(X_test, proba):
#     print(f"- {t} -> P(ham)={p[0]:.3f}, P(spam)={p[1]:.3f}")

samples = [
    "win free prize now",
    "project meeting tomorrow",
    "call now for limited offer",
    "deadline week schedule",
]

pred = model.predict(samples)
proba = model.predict_proba(samples)

# for text, p, label in zip(samples, proba, pred):
#     print(f"\n{text}")
#     print(f"P(ham)={p[0]:.3f}, P(spam)={p[1]:.3f} -> Predicted: {'SPAM' if label==1 else 'HAM'}")

tfidf_model = Pipeline(steps=[
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

tfidf_model.fit(X_train, y_train)

tfidf_pred = tfidf_model.predict(X_test)
tfidf_proba = tfidf_model.predict_proba(X_test)

print("\n=== TF-IDF + Naive Bayes ===")
print("X_test:", X_test)
print("Pred:", tfidf_pred)
print("True:", y_test)
print("Proba:", tfidf_proba)
print("cm:\n", confusion_matrix(y_test, tfidf_pred))