from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
