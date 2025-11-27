import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

# 1. Carregar os Dados
try:
    df = pd.read_csv('breast-cancer.csv', na_values='?')
except FileNotFoundError:
    print("Erro: O arquivo 'breast-cancer.csv' não foi encontrado. Certifique-se de que ele está no diretório correto.")
    exit()

print("✅ Dados Carregados e '?' substituído por NaN.")

# 2. Pré-processamento dos Dados
# a) Tratamento de valores ausentes (NaN)
for column in ['node-caps', 'breast-quad']:
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)

print("✅ Valores ausentes (NaN) preenchidos com a moda.")

# b) Conversão de variáveis categóricas para numéricas (One-Hot Encoding)
df['Class'] = df['Class'].replace({'no-recurrence-events': 0, 'recurrence-events': 1})
df['Class'] = df['Class'].astype('int')

X = df.drop('Class', axis=1)
y = df['Class']

# O 'drop_first=True' é importante para evitar a multicolinearidade
X_encoded = pd.get_dummies(X, drop_first=True)

print("✅ Variáveis categóricas convertidas (One-Hot Encoding).")

# 3. Divisão dos Dados
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)

print(f"✅ Dados divididos: Treino ({len(X_train)} amostras), Teste ({len(X_test)} amostras).")

# 4. Treinamento do Modelo Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

print("✅ Modelo Decision Tree treinado com sucesso.")

# 5. Avaliação do Modelo
y_pred = model.predict(X_test)

## 📊 Performance do Modelo

# Acurácia
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Resultados da Avaliação ---")
print(f"🎯 Acurácia do Modelo: {accuracy*100:.2f}%")

# Matriz de Contingência (Matriz de Confusão)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\n--- Matriz de Contingência (Matriz de Confusão) ---")
print(conf_matrix)

# Exibe a Matriz de Confusão em um formato mais legível (Tabela)
print("\n| Predito: 0 | Predito: 1 |")
print("|:---:|:---:|")
print(f"| {conf_matrix[0, 0]} (VP) | {conf_matrix[0, 1]} (FN) | **Real: 0 (Sem Recorrência)**")
print(f"| {conf_matrix[1, 0]} (FP) | {conf_matrix[1, 1]} (VP) | **Real: 1 (Com Recorrência)**")

## 🆕 Classificação de Novas Instâncias

# Exemplo 1: Caso Favorable (baixa chance de recorrência)
new_instance_favorable = pd.Series({
    'age': '50-59', 'menopause': 'ge40', 'tumor-size': '15-19', 'inv-nodes': '0-2',
    'node-caps': 'no', 'deg-malig': 1, 'breast': 'right', 'breast-quad': 'central',
    'irradiat': 'no'
})

# Exemplo 2: Caso Desfavorável (alta chance de recorrência)
new_instance_unfavorable = pd.Series({
    'age': '40-49', 'menopause': 'premeno', 'tumor-size': '35-39', 'inv-nodes': '12-14',
    'node-caps': 'yes', 'deg-malig': 3, 'breast': 'left', 'breast-quad': 'left_up',
    'irradiat': 'yes'
})

new_data = pd.DataFrame([new_instance_favorable, new_instance_unfavorable])

# Pré-processar as novas instâncias da mesma forma que os dados de treino
new_data_encoded = pd.get_dummies(new_data, drop_first=True)

# Garantir que as colunas das novas instâncias sejam idênticas às colunas de treino
missing_cols = set(X_train.columns) - set(new_data_encoded.columns)
# Adiciona as colunas ausentes (preenchendo com 0)
for c in missing_cols:
    new_data_encoded[c] = 0

# Reordena as colunas para coincidir com as colunas de treino (CRÍTICO!)
new_data_encoded = new_data_encoded[X_train.columns]

# Classificação
new_predictions = model.predict(new_data_encoded)
new_probabilities = model.predict_proba(new_data_encoded)

# Resultados
prediction_map = {0: 'Sem Recorrência (0)', 1: 'Com Recorrência (1)'}
print("\n--- Classificação de Novas Instâncias ---")

for i, pred in enumerate(new_predictions):
    prob_no = new_probabilities[i, 0]
    prob_yes = new_probabilities[i, 1]

    print(f"\nCaso {i+1} ({'Favorável' if i == 0 else 'Desfavorável'}):")
    print(f" -> Previsão: **{prediction_map[pred]}**")
    print(f" -> Probabilidades: Sem Recorrência ({prob_no:.2f}), Com Recorrência ({prob_yes:.2f})")