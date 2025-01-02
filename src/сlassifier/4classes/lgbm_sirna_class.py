
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
def remove_trailing_period(value):
    if isinstance(value, str) and value.endswith('.'):
        return value[:-1]
    return value
X = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/rdkit_right_desc.csv', index_col=0)
y = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/target_right_desc.csv', index_col=0)['0']
y = pd.DataFrame(y)
cons = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['siRNA concentration'].apply(remove_trailing_period).astype(float)
X['Concentration_nM'] = cons[cons <= 100]
X['Experiment_used_to_check_activity'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Experiment used to check activity'].astype(float)
X['Target_gene'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Target gene'].astype(float)
X['Cell_or_Organism_used'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Cell or Organism used'].astype(float)
X['Transfection_method'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Transfection method'].astype(float)
X['Duration_after_transfection'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Duration after transfection'].astype(float)
X['target'] = y
X = X.drop_duplicates()
X = X.reset_index(drop=True)
y = X['target']
X = X.drop(columns=['target'])



from sklearn.preprocessing import RobustScaler
def data_prep(X, y):
  X_scaled = X.drop(columns=['Concentration_nM',
                             'Target_gene',
                   'Cell_or_Organism_used',
                     'Transfection_method',
                     'Experiment_used_to_check_activity',
                     'Duration_after_transfection']).copy()

  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



  return X_train, X_test, y_train, y_test


# %%
def adjusted_r2_score(r2, n, p):
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

# %%
from sklearn.metrics import mean_squared_error, r2_score
def cross_val_score1(model, X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True)
    r2_scores = []
    adj_r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        n = X_test.shape[0]
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        r2_scores.append(r2)
        adj_r2_scores.append(adj_r2)

    return r2_scores

def categorize_value(value):
  if 0 <= value <= 25:
    return 1
  elif 25 < value <= 50:
    return 2
  elif 50 < value <= 75:
    return 3
  elif 75 < value <= 100:
    return 4
  else:
    return None

y = y.apply(categorize_value)

from sklearn.metrics import classification_report
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

X_train, X_valid, y_train, y_valid = data_prep(X, y)
# Создание модели LightGBM (классификатор)
model = lgb.LGBMClassifier()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тренировочном и валидационном наборах
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
# classification_report
print("Train:")
print(classification_report(y_train, y_train_pred))

print("\nValidation:")
print(classification_report(y_valid, y_valid_pred))

feature_importances = model.feature_importances_

# Создание списка кортежей (значение важности, имя признака)
top_features = sorted(zip(feature_importances, X_train.columns), key=lambda x: x[0], reverse=True)[:100]

# Извлечение имен топовых признаков
top_features_names = [feature[1] for feature in top_features]

# Фильтрация датафрейма по топовым признакам
X = X[top_features_names]

# Вычисление матрицы корреляции
corr_matrix = X.corr().abs()

# Выбор верхнего треугольника матрицы корреляции
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Поиск индексов колонок с корреляцией больше 0.95
high_cor = [column for column in upper.columns if any(upper[column] > 0.95)]  # Измените на 0.97, если нужно

# Исключение высококоррелирующих фич из списка топовых фич
features = [i for i in top_features_names if i not in high_cor]



# %%
print(features)

len(features)

X[features]

print('after selection')
print(len(features))

# %%
def remove_trailing_period(value):
    if isinstance(value, str) and value.endswith('.'):
        return value[:-1]
    return value
X = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/rdkit_right_desc.csv', index_col=0)[features]
y = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/target_right_desc.csv', index_col=0)['0']
y = pd.DataFrame(y)
cons = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['siRNA concentration'].apply(remove_trailing_period).astype(float)
X['Concentration_nM'] = cons[cons <= 100]
X['Experiment_used_to_check_activity'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Experiment used to check activity'].astype(float)
X['Target_gene'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Target gene'].astype(float)
X['Cell_or_Organism_used'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Cell or Organism used'].astype(float)
X['Transfection_method'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Transfection method'].astype(float)
X['Duration_after_transfection'] = pd.read_csv('/mnt/tank/scratch/igolovkin/siRNA/info_right_desc.csv')['Duration after transfection'].astype(float)
X['target'] = y
X = X.drop_duplicates()
X = X.reset_index(drop=True)
y = X['target']
X = X.drop(columns=['target'])
y = y.apply(categorize_value)
def data_prep(X, y):


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



  return X_train, X_test, y_train, y_test


X_train, X_valid, y_train, y_valid = data_prep(X, y)


# Создание модели LightGBM (классификатор)
model = lgb.LGBMClassifier()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тренировочном и валидационном наборах
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
# classification_report
print("Train:")
print(classification_report(y_train, y_train_pred))

print("\nValidation:")
print(classification_report(y_valid, y_valid_pred))

y

import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import optuna

def objective(trial):
    params = {
        'objective': 'multiclass',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
        'n_estimators': trial.suggest_int('n_estimators', 10, 1000),
        'max_depth': trial.suggest_int('max_depth', -1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0)
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)

    # Используйте метрику, которая лучше всего подходит для вашей задачи
    accuracy = accuracy_score(y_valid, y_pred)

    return 1 - accuracy  # Минимизируем ошибку

# Создаем изучатель
study = optuna.create_study(direction="minimize")

# Выполняем подбор параметров
study.optimize(objective, n_trials=100)  # Количество испытаний

# Лучшие параметры
best_params = study.best_params



# Создаем модель с лучшими параметрами
model = lgb.LGBMClassifier(**best_params)
model.fit(X_train, y_train)

# Предсказания на тренировочном, валидационном и тестовом наборах
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_valid)

# Вывод метрик для тренировочного, валидационного и тестового наборов
print("Тренировочный набор:")
print(classification_report(y_train, y_pred_train))
print(confusion_matrix(y_train, y_pred_train))


print("nТестовый набор:")
print(classification_report(y_valid, y_pred_test))
print(confusion_matrix(y_valid, y_pred_test))