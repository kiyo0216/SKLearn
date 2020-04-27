from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

x, y = datasets.load_digits(return_X_y=True)

svc = svm.SVC(kernel='linear')
C_s = np.logspace(-10, 0, 10)

scores = list()
k_fold = KFold(n_splits = 10)

print(cross_val_score(svc,x,y,cv=k_fold,n_jobs=-1))

plt.figure()
plt.show()
