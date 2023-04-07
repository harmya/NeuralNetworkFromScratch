import numpy as np
import matplotlib.pyplot as plt


a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
b = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

c = list(zip(a, b))

diff = [abs(a - b) for a, b in c]

plt.plot(diff)
plt.show()
