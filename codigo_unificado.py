#Contém todas as etapas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Carregar o dataset
df_train = pd.read_csv('train.csv')

# Pré-processamento dos dados
df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)
df_train.drop(columns=['Cabin'], inplace=True)

le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_train['Embarked'] = le.fit_transform(df_train['Embarked'])

# Separar variáveis independentes e alvo
X = df_train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df_train['Survived']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinamento do modelo
model = GaussianNB()
model.fit(X_train, y_train)

# Predição
y_pred = model.predict(X_test)

# Avaliação do modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Acurácia: {accuracy}")
print(f"Matriz de Confusão:\n{conf_matrix}")
print(f"Relatório de Classificação:\n{report}")

# Teste de predição
exemplo = pd.DataFrame([[1, 1, 25, 0, 0, 100, 0]], columns=X.columns)
predicao = model.predict(exemplo)
print("Sobreviveu" if predicao == 1 else "Não sobreviveu")
