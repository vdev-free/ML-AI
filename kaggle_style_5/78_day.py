import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1) Дані
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2) Вибір ознак та цілі
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# 3) Розбивка з однаковими пропорціями класів у train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# 4) Списки колонок за типом
numeric = ["Age", "Fare", "SibSp", "Parch"]   # суцільні числа
categorical = ["Sex", "Embarked", "Pclass"]   # категорії (Pclass як категорія)

# 5) Пайплайн для числових: імп'ютація медіаною -> стандартизація
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 6) Пайплайн для категорій: імп'ютація модою -> one-hot
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore"))  # не впаде на нових категоріях
])

# 7) Збірка всього разом
preprocess = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric),
    ("cat", categorical_pipeline, categorical),
])

# 8) Фінальний пайплайн: препроцесинг -> логістична регресія
model = Pipeline(steps=[
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=200))
])

# 9) Навчання
model.fit(X_train, y_train)

# 10) Передбачення і звіт
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, digits=3))

new_passenger = pd.DataFrame([{
    "Pclass": 3,        # 1/2/3 клас
    "Sex": "male",      # 'male'/'female'
    "Age": 28.0,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 7.925,
    "Embarked": "S"     # 'C','Q','S'
}])

print("Прогноз (0=не вижив, 1=вижив):", model.predict(new_passenger))
print("Ймовірність класу 1:", model.predict_proba(new_passenger)[:, 1])