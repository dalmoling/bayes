exemplo = pd.DataFrame([[1, 1, 25, 0, 0, 100, 0]], columns=X.columns)

predicao = model.predict(exemplo)
print("Sobreviveu" if predicao == 1 else "Não sobreviveu")
