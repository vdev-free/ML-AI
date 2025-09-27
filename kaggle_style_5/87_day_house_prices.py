import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

dataTrain = pd.read_csv('train_house_prices.csv')
dataTest = pd.read_csv('test_house_prices.csv')

X = dataTrain.drop('SalePrice', axis=1)
y = dataTrain['SalePrice']
X_test_final = dataTest.copy()

num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

num_transformer = Pipeline([
('imputer', SimpleImputer(strategy='median')),
('scaler', StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

models = {
'LinearRegression': LinearRegression(),
'RandomForestRegressor': RandomForestRegressor(n_estimators=200, random_state=42),
'GradientBoostingRegressor': GradientBoostingRegressor(n_estimators=300, random_state=42)
}

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    print(f"{name} RMSE: {rmse:.2f}")