# coding: utf-8
import numpy as np
import timeit
from sklearn import datasets, model_selection, metrics
from sklearn.neural_network import MLPClassifier

# fetch datasets under proxy
iris = datasets.load_iris()
print(iris)

iris_data = iris.data
iris_label = iris.target

data_train, data_test, label_train, label_test = model_selection.train_test_split(
    iris_data, iris_label, test_size=0.1, train_size=0.9)
print('train_size:', data_train.shape[0])
print('test_size:', data_test.shape[0])

tuned_parameters = [
    {
        "solver": ["lbfgs", "sgd", "adam"],
        "hidden_layer_sizes":[(100,), (100, 10), (100, 100, 10), (100, 100, 100, 10), (100, 100, 100, 100, 10)],
    }
]
licv = model_selection.GridSearchCV(
    MLPClassifier(early_stopping=True),
    param_grid=tuned_parameters,
    scoring="accuracy"
)
print('Learning Time(s):', timeit.timeit(
    lambda: licv.fit(data_train, label_train), number=1))
print('licv.best_params_ : ', licv.best_params_)
print('licv.best_estimator_ : ', licv.best_estimator_)
print('licv.best_score_ : ', licv.best_score_)

pre = licv.predict(data_test)
ac_score = metrics.accuracy_score(label_test, pre)
print('accuracy score:', ac_score)

co_mat = metrics.confusion_matrix(label_test, pre)
print('confusion matrix:')
print(co_mat)
cl_repo = metrics.classification_report(label_test, pre)
print('classification report:')
print(cl_repo)
