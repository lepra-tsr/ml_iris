from sklearn import svm, datasets, metrics

model = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
model.fit(X[:-1], y[:-1])
prediction = model.predict([X[-1]])
print(X[-1], prediction, y[-1], metrics.accuracy_score([y[-1]], prediction))
