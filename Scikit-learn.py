from sklearn import datasets, neighbors, linear_model
from sklearn.neighbors import KNeighborsClassifier

x_digits, y_digits = datasets.load_digits(return_X_y=True)
x_digits = x_digits / x_digits.max()

print(x_digits.shape)

val_size = int(x_digits.shape[0] / 10)
x_train = x_digits[:val_size]
x_test = x_digits[val_size:]
y_train = y_digits[:val_size]
y_test = y_digits[val_size:]

KNN = KNeighborsClassifier()
KNN.fit(x_train, y_train)
print(KNN.predict(x_test))
print(y_test)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
print(KNN.predict(x_test))
print(y_test)
