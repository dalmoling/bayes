import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_train = pd.read_csv('train.csv')

df_train['Age'].fillna(df_train['Age'].median(), inplace=True)
df_train['Embarked'].fillna(df_train['Embarked'].mode()[0], inplace=True)

df_train.drop(columns=['Cabin'], inplace=True)

le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_train['Embarked'] = le.fit_transform(df_train['Embarked'])

df_train.head()
