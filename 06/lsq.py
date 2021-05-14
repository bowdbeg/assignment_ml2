import numpy as np
import matplotlib.pyplot as plt


lmd = 8


def solve(x, y, lmd=0):
    w = np.matmul(x.T, x) + lmd / 2 * np.eye(x.shape[-1])
    print("mul")
    print(w)
    print("det")
    print(np.linalg.det(w))
    w = np.linalg.inv(w)
    print("inv")
    print(w)
    w = np.matmul(np.matmul(w, x.T), y)
    print("ans")
    print(w)
    return w


data = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, 2.0], [-2.0, -2.0],[0.0, 0.0], [0.0, 5.0]])

data1 = data
x1 = np.vstack([np.ones(data1.shape[0]), data1[:, 0]]).T
y1 = data1[:, 1]
print(1)

w1 = solve(x1, y1)

# plt.scatter(x1[:, 1], y1, c="r", marker="o", s=100, label="(1)")

####

data2 = data
x2 = np.vstack([np.ones(data2.shape[0]), data2[:, 0]]).T
y2 = data2[:, 1]
print(2)
w2 = solve(x2, y2, lmd=lmd)

y2_ = y2
# y2_[-1] = 5.0
plt.scatter(x2[:, 1], y2_, c="b", marker="x", s=100, label="data")

plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc="upper left")

plt.savefig("./06_scatter.pdf")

x_ = np.array([[1, -5], [1, 5]])

print(np.matmul(x_, w1))
p = plt.plot(x_[:, 1], np.matmul(x_, w1), linestyle="dashed", color="m", label="(1)")
p = plt.plot(x_[:, 1], np.matmul(x_, w2), linestyle="dashed", color="c", label="(2)")
# p = plt.plot([0, 0], x_[:, 1], linestyle="dotted", color="c", label="(2) x = 0")
plt.legend(loc="upper left")

# plt.show()
plt.savefig("./06_answer.pdf")
