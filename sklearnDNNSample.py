# coding: utf-8
import numpy as np
import timeit
from sklearn import datasets, model_selection, metrics
from sklearn.neural_network import MLPClassifier

# fetch datasets under proxy
# python sklearnDNNSample.py --proxy="$P_HOST:$P_PORT"
mnist = datasets.fetch_mldata('MNIST original', data_home='data/mnist/')

mnist_data = mnist.data / 255
mnist_label = mnist.target

data_train, data_test, label_train, label_test = model_selection.train_test_split(
    mnist_data, mnist_label, test_size=0.05, train_size=0.3)  # 学習用は全データの30%、テスト用は5%に設定
print('train_size:', data_train.shape[0])
print('test_size:', data_test.shape[0])

classifiers = [
    (('neural_network Adam', MLPClassifier(
        solver="adam", early_stopping=True, verbose=True))),
    (('neural_network Adam(2 layer)', MLPClassifier(
        solver="adam", early_stopping=True, hidden_layer_sizes=(100, 10)))),
    (('neural_network Adam(3 layer)', MLPClassifier(solver="adam",
                                                    early_stopping=True, hidden_layer_sizes=(100, 100, 10)))),
    (('neural_network Adam(4 layer)', MLPClassifier(solver="adam",
                                                    early_stopping=True, hidden_layer_sizes=(100, 100, 100, 10))))
]

classifier_names = [
    'neural_network Adam',
    'neural_network Adam(2 layer)',
    'neural_network Adam(3 layer)',
    'neural_network Adam(4 layer)'
]

tuned_parameters = [
    {
        "solver": ["lbfgs", "sgd", "adam"],
        "hidden_layer_sizes":[(100,), (100, 10), (100, 100, 10), (100, 100, 100, 10)],
    }
]
licv = model_selection.GridSearchCV(
    MLPClassifier(early_stopping=True),
    param_grid=tuned_parameters,
    scoring="accuracy"
)
print('Learning Time(s):', timeit.timeit(lambda: licv.fit(data_train, label_train), number=1))
print('licv.best_params_ : ', licv.best_params_)
print('licv.best_estimator_ : ', licv.best_estimator_)
print('licv.best_score_ : ', licv.best_score_)

exit()


for name, clf in classifiers:
    print(name, 'start')

    print('learning Time(s):', timeit.timeit(
        lambda: clf.fit(data_train, label_train), number=1))
    pre = clf.predict(data_test)
    ac_score = metrics.accuracy_score(label_test, pre)
    print('accuracy score:', ac_score)

    co_mat = metrics.confusion_matrix(label_test, pre)
    print('confusion matrix:')
    print(co_mat)
    cl_repo = metrics.classification_report(label_test, pre)
    print('classification report:')
    print(cl_repo)

    print(name, 'end')
