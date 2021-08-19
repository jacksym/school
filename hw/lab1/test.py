import numpy as np
import matplotlib.pyplot as plt

array1 = np.array([10, 20, 30, 40, 50, 60])
array2 = np.array([20, 50, 100, 150, 70, 80])
array3 = np.array([1, 2, 3, 4, 5, 6])
extrapeaks = np.array([50, 20])

#array = np.array(array1, ndmin=2)
array = np.empty([1,6])
print(array)
array = np.append(array, [array3, array1], axis=0)
print(array)

x = np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
print(x)

list = []
list.append(array1)
list.append(array2)
narray = np.array(list)
print(narray)
