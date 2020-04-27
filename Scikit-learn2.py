from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
x = iris.data
y = iris.target

x = x[y != 0, :2]
y = y[y != 0]

val_data = int(0.9 * len(x))
x_train = x[:val_data]
x_test = x[val_data:]
y_train = y[:val_data]
y_test = y[val_data:]


SVC = svm.SVC(kernel='linear')
SVC.fit(x_train, y_train)
print(SVC.predict(x_test))
print(y_test)