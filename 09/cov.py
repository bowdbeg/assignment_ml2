import numpy as np
import matplotlib.pyplot as plt

def solve(x):
    print("matrix")
    print(x)
    print("det")
    print(np.linalg.det(x))
    x = np.linalg.inv(x)
    print("inv")
    print(x)
    return x

mu = np.array([1,1])
sigma = np.array([[1,2],[2,1]])

eigval,eigvec = np.linalg.eig(sigma)

print('eigval')
print(eigval)
print('eigvec')
print(eigvec)

assert np.dot(eigvec[0], eigvec[1]) < 1e-3

x = np.arange(-1,1,0.001)
y = np.concatenate([np.sqrt(1-x**2), np.flip(-np.sqrt(1-x**2))])
x = np.concatenate([x,np.flip(x)])

# shape
coord = np.stack([x,y])
coord = np.stack([eigval]*coord.shape[-1],axis=-1) * coord / np.sqrt(2)

# rotation
edge = np.linalg.norm(eigvec[0])
sin = eigvec[0,0] / edge 
cos = eigvec[0,1]/edge
rotmat=np.array([[cos,-sin],[sin,cos]])
coord = np.matmul(rotmat,coord)

# position
coord = coord + np.stack([mu]*coord.shape[-1], axis=-1)

x,y = coord

plt.figure(figsize=(5,5))
# plt.plot(x,y)

plt.xlim(-1,3)
plt.ylim(-1,3)
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid()

# plt.show()
plt.savefig('09.pdf')