import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression

data = load_breast_cancer()

X = data.data
y = data.target

# print("Total:", X.shape, y.shape)
# print("Total class balance:", np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# print("\nTrain:", X_train.shape, y_train.shape)
# print("Test:", X_test.shape, y_test.shape)

# print("\nTrain class balance:", np.unique(y_train, return_counts=True))
# print("Test  class balance:", np.unique(y_test, return_counts=True))

X_scaled_bad = StandardScaler().fit_transform(X)

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_scaled_bad, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf_bad = LogisticRegression(max_iter=2000)
clf_bad.fit(X_train_b, y_train_b)

bad_test_acc = accuracy_score(y_test_b, clf_bad.predict(X_test_b))
# print("\nBAD leakage test acc:", bad_test_acc)

# --- OK (NO LEAKAGE): scaler fitted ONLY on train ---
scaler_ok = StandardScaler()
X_train_scaled_ok = scaler_ok.fit_transform(X_train)  # fit тільки на train
X_test_scaled_ok = scaler_ok.transform(X_test)        # transform на test тим самим scaler

clf_ok = LogisticRegression(max_iter=2000)
clf_ok.fit(X_train_scaled_ok, y_train)

ok_test_acc = accuracy_score(y_test, clf_ok.predict(X_test_scaled_ok))
# print("OK no-leakage test acc:", ok_test_acc)

pipe = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000))
])

pipe.fit(X_train, y_train)

pipe_test_acc = accuracy_score(y_test, pipe.predict(X_test)) 
# print("\nPIPELINE test acc:", pipe_test_acc)

print("\n--- TIME SPLIT DEMO ---")
# 1) Синтетичні дані: день -> попит
n = 200
rng = np.random.RandomState(42)

days = np.arange(n)  # 0..199 (час)
trend = days * 0.5   # тренд: попит росте з часом
noise = rng.normal(0, 10, size=n)
y_ts = trend + noise

X_ts = days.reshape(-1, 1)  # ознака: лише "день"

# 2) ПОГАНО: random split (перемішали майбутнє з минулим)
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_ts, y_ts, test_size=0.2, random_state=42, shuffle=True
)
lr = LinearRegression()
lr.fit(X_tr_r, y_tr_r)
pred_r = lr.predict(X_te_r)
mae_random = mean_absolute_error(y_te_r, pred_r)
print("MAE with RANDOM split:", round(mae_random, 2))

# 3) ПРАВИЛЬНО: time split (train = минуле, test = майбутнє)
split_point = int(n * 0.8)
X_tr_t, X_te_t = X_ts[:split_point], X_ts[split_point:]
y_tr_t, y_te_t = y_ts[:split_point], y_ts[split_point:]

lr2 = LinearRegression()
lr2.fit(X_tr_t, y_tr_t)
pred_t = lr2.predict(X_te_t)
mae_time = mean_absolute_error(y_te_t, pred_t)
print("MAE with TIME split:", round(mae_time, 2))

print("\nRandom split: min day in train/test:", X_tr_r.min(), X_te_r.min())
print("Random split: max day in train/test:", X_tr_r.max(), X_te_r.max())

print("\nTime split:   min day in train/test:", X_tr_t.min(), X_te_t.min())
print("Time split:   max day in train/test:", X_tr_t.max(), X_te_t.max())


# --- LEAKAGE CHECKLIST (quick) ---
# 1) Будь-який preprocessing (scaler/encoder/PCA/feature selection) -> fit ТІЛЬКИ на train
# 2) Target encoding / mean encoding -> дуже легко дає leakage (обережно!)
# 3) Агрегації типу "середнє по всім даним" -> роби в рамках train, або через CV
# 4) Дублі/майже дублі між train і test -> прибирай (особливо в текстах/зображеннях)
# 5) Time series -> НЕ shuffle. Тільки time split / backtesting
# 6) Один користувач у train і test -> leakage (треба GroupKFold / груповий split)
# 7) Feature з майбутнього ("покупки за наступні 7 днів") -> це пряме leakage
# 8) Підбір порогу/гіперпараметрів на test -> test більше не чесний (треба validation)
# 9) Нормалізація/імпутація на всіх даних до split -> leakage
# 10) Якщо метрики “занадто ідеальні” -> підозра на leakage, перевіряй пайплайн
