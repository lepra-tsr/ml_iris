import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("./iris.csv", delimiter=",", header=0)
fig = iris[iris.Species == 'Iris-setosa'].plot(
    kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='orange', label='Setosa')
iris[iris.Species == 'Iris-versicolor'].plot(
    kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='blue', label='versicolor', ax=fig)
iris[iris.Species == 'Iris-virginica'].plot(
    kind='scatter', x='PetalLengthCm', y='PetalWidthCm', color='green', label='virginica', ax=fig)
fig.set_xlabel("Petal Length")
fig.set_ylabel("Petal Width")
fig.set_title("Petal Length VS Width")
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.show()
