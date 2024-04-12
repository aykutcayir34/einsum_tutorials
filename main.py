import numpy as np

def rand(*args):
    return np.random.rand(*args)

print(np.matmul(rand(4, 5), rand(5, 6)).shape)

A = rand(5, 32)
B = rand(10, 32)

print(np.matmul(A, B.T).shape)

A = rand(10, 4, 5)
B = rand(10, 5, 6)

print(np.matmul(A, B).shape)

E = 32
query_embedding = rand(E)
key_embedding = rand(10, E)

v1 = np.matmul(query_embedding, key_embedding.T)
v2 = np.matmul(key_embedding, query_embedding)

print(v1.shape, v2.shape, np.allclose(v1, v2))

N = 100
E = 32

query_embedding = rand(N, E)
key_embedding = rand(N, 10, E)

v1 = np.matmul(query_embedding.reshape(N, 1, E), key_embedding.transpose(0, 2, 1))
print(v1.shape)

#einsum examples
A, B = rand(4, 5), rand(5, 6)
m = np.matmul(A, B)
e = np.einsum("ij,jk->ik", A, B)
print(np.allclose(m, e))

A, B = rand(5, 32), rand(10, 32)
e = np.einsum("ij,kj->ik", A, B)
m = np.matmul(A, B.T)
print(np.allclose(m, e))

A, B = rand(10, 4, 5), rand(10, 5, 6)
e = np.einsum("ijk,ikm->ijm", A, B)
m = np.matmul(A, B)
print(np.allclose(m, e), m.shape, e.shape)

query_embedding = rand(E)
key_embedding = rand(10, E)
v1 = np.einsum("i,li->l", query_embedding, key_embedding)
v2 = np.matmul(query_embedding, key_embedding.T)
print(v1.shape, v2.shape, np.allclose(v1, v2))

N = 100 
E = 32
query_embedding = rand(N, E)
key_embedding = rand(N, 10, E)
v1 = np.squeeze(np.matmul(query_embedding.reshape(N, 1, E), key_embedding.transpose(0, 2, 1)))
v2 = np.einsum("ij,ikj->ik", query_embedding, key_embedding)
print(v1.shape, v2.shape, np.allclose(v1, v2))