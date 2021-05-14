import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, -1.0], [1.0, -10000.0, 0.0]])
y = np.array([1, 0, 1, 0, 1])
# x = np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, -10.0, 0.0]])
# y = np.array([1, 0, 1])

w = np.matmul(np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T), y)
print((np.matmul(x.T, x)))
print(np.linalg.det((np.matmul(x.T, x))))
print((np.linalg.inv(np.matmul(x.T, x))))
print(w)

x1 = x[:-1]
y1 = y[:-1]
w1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(x1.T, x1)), x1.T), y1)
print((np.matmul(x1.T, x1)))
print((np.linalg.inv(np.matmul(x1.T, x1))))
print(w1)

x_p = np.array([a for a, b in zip(x, y) if b])
x_p = np.vstack([x_p, np.array([[1,-3,0]])])
print(x_p)
x_n = np.array([a for a, b in zip(x, y) if not b])

plt.scatter(x_p[:, 1], x_p[:, 2], c="r", marker="o",s=100)
plt.scatter(x_n[:, 1], x_n[:, 2], c="b", marker="x",s=100)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xlabel("x_1")
plt.ylabel("x_2")

plt.savefig('./03_scatter.pdf')

x_ = np.array([-11, 3])


p = plt.plot(x_, -(w[0] + w[1] * x_ - 0.5) / w[-1], linestyle="dashed", color="c", label='All')
p1 = plt.plot(x_, -(w1[0] + w1[1] * x_ - 0.5) / w1[-1], linestyle="dotted", color="m", label='w/o ID5')
plt.legend()

# plt.show()
plt.savefig('./03_answer.pdf')
