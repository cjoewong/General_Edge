from LinearRegression import LinearRegression
import numpy as np

lr = LinearRegression([])
X = np.array([[1,2,3],[2,3,5]])
y = np.array([1.5,2.5])[:,None]
w, b = lr.run(X, y)
print(X@w+b)
print(w)
print(b)
