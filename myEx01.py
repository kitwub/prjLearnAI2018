import numpy as np

# preparing
rng = np.random.RandomState(123)
d = 2
N = 10
mean = 5

#
def step_func(x):
    return 1 if x > 0 else 0

def y(x):
    return step_func(np.dot(w, x) + b)

def t(i):
    return 0 if i < N else 1


p = np.array([0, 0]) + rng.randn(N, d)
n = np.array([mean, mean]) + rng.randn(N, d)
x = np.concatenate((p, n), axis=0)

# w = rng.randn(d).reshape(1, d)
# b = rng.randn(1)
w = np.zeros(d)
b = np.zeros(1)

d_w = np.zeros(d)
d_b = np.zeros(1)
L = np.zeros(1)

eps = 0.00001
counter = 1

while True:
    print(counter)
    counter += 1

    end_flag = True

    for i in range(N * 2):
        L = t(i) - y(x[i])
        d_w = L * x[i]
        d_b = L

        w += d_w
        b += d_b

        end_flag = end_flag and (all(d_w < eps) and (d_b < eps))
        pass

    if end_flag:
        break

print(w, b)
