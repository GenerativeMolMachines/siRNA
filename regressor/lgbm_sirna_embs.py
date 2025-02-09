# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# %%
data_desc = pd.read_csv('E:\My_projects\gene_embeddings__descriptors.csv')

# %%
import json
with open("E:\\My_projects\\gene_mistral_common.json", "r") as f:
    embeddings_data = json.load(f)

# Извлечение эмбеддингов
data = []
for item in embeddings_data:
    input_value = item['input']
    embedding_values = item['embedding']

    # Разбиваем embedding на отдельные значения
    embedding_list = []  # Создаем пустой список для значений эмбеддинга
    for value in embedding_values:  # Итерируем по элементам списка embedding_values
        embedding_list.append(float(value))  # Добавляем значения в список embedding_list

    # Создаем словарь для записи в DataFrame
    row_data = {'gene_sequence': input_value}
    for i, value in enumerate(embedding_list):
        row_data[f'embedding_{i+1}'] = value  # Добавляем значения в словарь row_data

    data.append(row_data)

# Создаем DataFrame
embs = pd.DataFrame(data)
embs
embs

# %%
# Задаём имя колонки с генами
gene_column = 'gene_sequence'

# Проверяем наличие столбца gene в обоих датафреймах
if gene_column not in data_desc.columns:
    raise ValueError(f"Столбец '{gene_column}' не найден в первом датафрейме.")
if gene_column not in embs.columns:
    raise ValueError(f"Столбец '{gene_column}' не найден во втором датафрейме.")

# Проверяем уникальность столбца gene во втором датафрейме
if not embs[gene_column].is_unique:
    raise ValueError(f"Столбец '{gene_column}' во втором датафрейме не уникален. Убедитесь, что каждому значению гена соответствует только один набор эмбеддингов.")


# Слияние датафреймов по столбцу gene
merged_df = pd.merge(data_desc, embs, on=gene_column, how='left')

# Проверка на наличие NaN значений
if merged_df.isnull().any().any():
     na_count = merged_df.isnull().sum().sum()
     print(f"Предупреждение: {na_count} значения NaN были обнаружены в объединенном датасете. Проверьте, все ли значения генов из data1.csv присутствуют в data2.csv.")

# %%
def remove_trailing_period(value):
    if isinstance(value, str) and value.endswith('.'):
        return value[:-1]
    return value
merged_df['Concentration, nM'] = merged_df['Concentration, nM'].apply(remove_trailing_period).astype(float)
merged_df = merged_df[merged_df['Concentration, nM'] <= 100]
y = merged_df['Efficacy, %']
X = merged_df.drop(columns=['Efficacy, %', 'siRNA concentration', 'Unnamed: 0',	'SMDBid', 'Target gene', 'gene_sequence'])
X



# %%
X = X.rename(columns={'Concentration, nM': 'Concentration_nM', 'Cell or Organism used': 'Cell_or_Organism_used', 'Transfection method': 'Transfection_method', 
                      'Experiment used to check activity': 'Experiment_used_to_check_activity', 'Duration after transfection': 'Duration_after_transfection'})

# %%
from sklearn.preprocessing import RobustScaler
def data_prep(X, y):
  X_scaled = X.drop(columns=['Concentration_nM',
                   'Cell_or_Organism_used',
                     'Transfection_method',
                     'Experiment_used_to_check_activity',
                     'Duration_after_transfection']).copy()


#  scalers = {}  # Словарь для хранения scaler'ов для каждого столбца

#  for column in X.drop(columns=['Concentration_nM',
#                             'Target_gene',
#                   'Cell_or_Organism_used',  
#                     'Transfection_method',
#                     'Experiment_used_to_check_activity',
#                     'Duration_after_transfection']).columns:
#    scaler = RobustScaler()
#    X_scaled[column] = scaler.fit_transform(X[[column]])

#    scalers[column] = scaler
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

# %%
import lightgbm as lgb
import lightgbm
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns



X_train, X_valid, y_train, y_valid = data_prep(X, y)
# Создание модели LightGBM
model = lgb.LGBMRegressor()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тренировочном и валидационном наборах
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)

# Метрики
print("Train:")
print(f"RMSE: {mean_squared_error(y_train, y_train_pred, squared=False)}")
print(f"R^2: {r2_score(y_train, y_train_pred)}")

print("\nValidation:")
print(f"RMSE: {mean_squared_error(y_valid, y_valid_pred, squared=False)}")
print(f"R^2: {r2_score(y_valid, y_valid_pred)}")

# Кросс-валидация
r2_mean= cross_val_score1(model, X, y)
print("\nCross-Validation:")
print(f"R2 Score: {r2_mean}")


# %%
X_train

# %%
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


# %%
len(features)

# %%
X[features]

# %%
print('after selection')
print(len(features))


def remove_trailing_period(value):
    if isinstance(value, str) and value.endswith('.'):
        return value[:-1]
    return value
data_desc = pd.read_csv('E:\My_projects\gene_embeddings__descriptors.csv')

import json
with open("E:\\My_projects\\gene_mistral_common.json", "r") as f:
    embeddings_data = json.load(f)

# Извлечение эмбеддингов
data = []
for item in embeddings_data:
    input_value = item['input']
    embedding_values = item['embedding']

    # Разбиваем embedding на отдельные значения
    embedding_list = []  # Создаем пустой список для значений эмбеддинга
    for value in embedding_values:  # Итерируем по элементам списка embedding_values
        embedding_list.append(float(value))  # Добавляем значения в список embedding_list

    # Создаем словарь для записи в DataFrame
    row_data = {'gene_sequence': input_value}
    for i, value in enumerate(embedding_list):
        row_data[f'embedding_{i+1}'] = value  # Добавляем значения в словарь row_data

    data.append(row_data)

# Создаем DataFrame
embs = pd.DataFrame(data)
embs

# Задаём имя колонки с генами
gene_column = 'gene_sequence'

# Проверяем наличие столбца gene в обоих датафреймах
if gene_column not in data_desc.columns:
    raise ValueError(f"Столбец '{gene_column}' не найден в первом датафрейме.")
if gene_column not in embs.columns:
    raise ValueError(f"Столбец '{gene_column}' не найден во втором датафрейме.")

# Проверяем уникальность столбца gene во втором датафрейме
if not embs[gene_column].is_unique:
    raise ValueError(f"Столбец '{gene_column}' во втором датафрейме не уникален. Убедитесь, что каждому значению гена соответствует только один набор эмбеддингов.")


# Слияние датафреймов по столбцу gene
merged_df = pd.merge(data_desc, embs, on=gene_column, how='left')

# Проверка на наличие NaN значений
if merged_df.isnull().any().any():
     na_count = merged_df.isnull().sum().sum()
     print(f"Предупреждение: {na_count} значения NaN были обнаружены в объединенном датасете. Проверьте, все ли значения генов из data1.csv присутствуют в data2.csv.")

merged_df['Concentration, nM'] = merged_df['Concentration, nM'].apply(remove_trailing_period).astype(float)
merged_df = merged_df[merged_df['Concentration, nM'] <= 100]
y = merged_df['Efficacy, %']
conc = merged_df['Concentration, nM']
cell = merged_df['Cell or Organism used']
transf = merged_df['Transfection method']
exper = merged_df['Experiment used to check activity']
dura = merged_df['Duration after transfection']
X = merged_df.drop(columns=['Efficacy, %', 'siRNA concentration', 'Unnamed: 0',	'SMDBid', 'Target gene', 'gene_sequence'])[features]
X = X.rename(columns={'Concentration, nM': 'Concentration_nM', 'Cell or Organism used': 'Cell_or_Organism_used', 'Transfection method': 'Transfection_method', 
                      'Experiment used to check activity': 'Experiment_used_to_check_activity', 'Duration after transfection': 'Duration_after_transfection'})
X['Concentration_nM'] = conc
X['Cell_or_Organism_used'] = cell
X['Transfection_method'] = transf
X['Experiment_used_to_check_activity'] = exper
X['Duration_after_transfection'] = dura
def data_prep(X, y):


  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



  return X_train, X_test, y_train, y_test

X_train, X_valid, y_train, y_valid = data_prep(X, y)
# Создание модели LightGBM
model = lgb.LGBMRegressor()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тренировочном и валидационном наборах
y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)


# Метрики
print('Metrics after feature selection')
print("Train:")
print(f"RMSE: {mean_squared_error(y_train, y_train_pred, squared=False)}")
print(f"R^2: {r2_score(y_train, y_train_pred)}")

print("\nValidation:")
print(f"RMSE: {mean_squared_error(y_valid, y_valid_pred, squared=False)}")
print(f"R^2: {r2_score(y_valid, y_valid_pred)}")

# Кросс-валидация
r2_mean= cross_val_score1(model, X, y)
print("\nCross-Validation:")
print(f"=R2 Score: {r2_mean}")


# %%
merged_df['Concentration, nM'] = merged_df['Concentration, nM'].apply(remove_trailing_period).astype(float)
merged_df = merged_df[merged_df['Concentration, nM'] <= 100]
y = merged_df['Efficacy, %']
conc = merged_df['Concentration, nM']
cell = merged_df['Cell or Organism used']
transf = merged_df['Transfection method']
exper = merged_df['Experiment used to check activity']
dura = merged_df['Duration after transfection']
X = merged_df.drop(columns=['Efficacy, %', 'siRNA concentration', 'Unnamed: 0',	'SMDBid', 'Target gene', 'gene_sequence'])[features]
X = X.rename(columns={'Concentration, nM': 'Concentration_nM', 'Cell or Organism used': 'Cell_or_Organism_used', 'Transfection method': 'Transfection_method', 
                      'Experiment used to check activity': 'Experiment_used_to_check_activity', 'Duration after transfection': 'Duration_after_transfection'})
X['Concentration_nM'] = conc
X['Cell_or_Organism_used'] = cell
X['Transfection_method'] = transf
X['Experiment_used_to_check_activity'] = exper
X['Duration_after_transfection'] = dura
X['target'] = y
X = X.drop_duplicates()
y = X['target']
X = X.drop(columns=['target'])
print('After hyperparams tuning')
X_train, X_valid, y_train, y_valid = data_prep(X, y)
print('Train:', len(X_train))
print('Valid:', len(X_valid), end='\n\n')

# %%
def objective(trial, data, target):

    train_x, test_x, train_y, test_y = data_prep(data, target)

    param = {
        'metric': 'rmse',
        'n_estimators' : trial.suggest_int('n_estimators', 100, 20000),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006, 0.008, 0.01, 0.014, 0.017, 0.02]),
        'max_depth': trial.suggest_int('max_depth', 3, 100),
        'num_leaves': trial.suggest_int('num_leaves', 2, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth': trial.suggest_int('min_data_per_groups', 1, 100),
        'max_bin': trial.suggest_int('max_bin', 10, 255),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-5, 10.0),
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
    }
    model = lgb.LGBMRegressor(**param)

    model.fit(train_x,train_y,eval_set=[(test_x,test_y)])

    preds = model.predict(test_x)

    r2 = metrics.r2_score(test_y, preds)

    return r2



study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)


params=study.best_params
params['metric'] = 'rmse'
params['cat_smooth'] = params.pop('min_data_per_groups')

print('After hyperparams tuning')


train_data = lgb.Dataset(X_train, y_train)
valid_data = lgb.Dataset(X_valid, y_valid, reference=train_data)

data1 = lgb.Dataset(X, label=y)


print('Starting training...')

# train
gbm = lgb.train(params,
                train_data,
                num_boost_round=10000,
                valid_sets=[valid_data])
print()

# save model to file
print('Saving model...')
gbm.save_model('model.txt')

# predict
print('Starting predicting...')
y_pred = gbm.predict(X_valid, num_iteration=gbm.best_iteration)

# %%
from matplotlib.patches import Patch
y_train_pred = gbm.predict(X_train, num_iteration=gbm.best_iteration)
r2_test = metrics.r2_score(y_valid, y_pred)
MAE_test = metrics.mean_absolute_error(y_valid, y_pred)
MSE_test = metrics.mean_squared_error(y_valid, y_pred)
RMSE_test = np.sqrt(metrics.mean_squared_error(y_valid, y_pred))
r2_train = metrics.r2_score(y_train, y_train_pred)
MAE_train = metrics.mean_absolute_error(y_train, y_train_pred)
MSE_train = metrics.mean_squared_error(y_train, y_train_pred)
RMSE_train = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
n_test = len(y_valid)
p_test = X_valid.shape[1]
adj_r2_test = adjusted_r2_score(r2_test, n_test, p_test)

n_train = len(y_train)
p_train = X_train.shape[1]
adj_r2_train = adjusted_r2_score(r2_train, n_train, p_train)

print('r2_test:', r2_test)
print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('Adjusted r2_test:', adj_r2_test)

# %%
print('r2_train:', r2_train)
print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('Adjusted r2_train:', adj_r2_train)
print(X_train.shape)
real_patch = Patch(color='#DD7059', label='train values')
pred_patch = Patch(color='#569FC9', label='test values')
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", rc=custom_params)
f, ax = plt.subplots(figsize=(16, 12))
plt.scatter(y_train, y_train_pred, color='#DD7059', s=70)
plt.scatter(y_valid, y_pred, color='#569FC9',s=70)
plt.plot(y_valid, y_valid, color='gray')
plt.legend(handles=[real_patch, pred_patch, plt.Line2D([], [], color='gray', label='Ideal line')])
plt.title('')
plt.xlabel('true')
plt.ylabel('predict')
plt.savefig('lgbm_tuned_nikita_final_restore1.png')

# %%
print('r2_test:', r2_test)
print('MAE_test:', MAE_test)
print('MSE_test:', MSE_test)
print('RMSE_test:', RMSE_test)
print('Adjusted r2_test:', adj_r2_test)

print('r2_train:', r2_train)
print('MAE_train:', MAE_train)
print('MSE_train:', MSE_train)
print('RMSE_train:', RMSE_train)
print('Adjusted r2_train:', adj_r2_train)


