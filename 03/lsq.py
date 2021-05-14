import numpy as np
import matplotlib.pyplot as plt

data1 = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0], [4.0, 4.0], [3.0, 4.0]])
x1 = np.vstack([np.ones(data1.shape[0]), data1[:, 0]]).T
y1 = data1[:, 1]
print(1)

w1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(x1.T, x1)), x1.T), y1)
print((np.matmul(x1.T, x1)))
print(np.linalg.det((np.matmul(x1.T, x1))))
print((np.linalg.inv(np.matmul(x1.T, x1))))
print(np.matmul(np.linalg.inv(np.matmul(x1.T, x1)), x1.T))
print(w1)

plt.scatter(x1[:, 1], y1, c="r", marker="o", s=100, label="(1)")

####

data2 = np.array([[0.0, 4.0], [1.0, 0.0], [0.0, -4.0], [-1.0, 0.0]])
x2 = np.vstack([np.ones(data2.shape[0]), data2[:, 0]]).T
y2 = data2[:, 1]
print(2)
w2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(x2.T, x2)), x2.T), y2)
print((np.matmul(x2.T, x2)))
print(np.linalg.det((np.matmul(x2.T, x2))))
print((np.linalg.inv(np.matmul(x2.T, x2))))
print(np.matmul(np.linalg.inv(np.matmul(x2.T, x2)), x2.T))
print(w2)

plt.scatter(x2[:, 1], y2, c="b", marker="x", s=100, label="(2)")

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel("x_1")
plt.ylabel("x_2")
plt.legend(loc="upper left")

plt.savefig("./03_scatter.pdf")

x_ = np.array([[1, -5], [1, 5]])

print(np.matmul(x_, w1))
p = plt.plot(x_[:, 1], np.matmul(x_, w1), linestyle="dashed", color="m", label="(1)")
p = plt.plot(x_[:, 1], np.matmul(x_, w2), linestyle="dashed", color="c", label="(2) y = 0")
p = plt.plot([0, 0], x_[:, 1], linestyle="dotted", color="c", label="(2) x = 0")
plt.legend(loc="upper left")

# plt.show()
plt.savefig("./03_answer.pdf")
