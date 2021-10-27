import torch


def sparse_softmax(sparse, dim):
    """Pure Python softmax of a sparse tensor. Assuming -inf for
    unspecified sparse tensor data. This is a prototype of
    sparse softmax algorithm in Python.
    """
    dtype = sparse.dtype
    device = sparse.device
    # softmax is non-linear operation, so sparse tensors must
    # be coalesced.
    sparse = sparse.coalesce()
    print("sparse", sparse)
    inf = float('inf')
    indices = sparse._indices()
    values = sparse._values()
    if dim < sparse.sparse_dim():
        nnz = sparse._nnz()
        # compute pool indices
        size = sparse.size()
        strides = torch.ones((sparse.sparse_dim(), 1), dtype=indices.dtype, device=indices.device)
        for i in reversed(range(sparse.sparse_dim() - 1)):
            strides[i, 0] = strides[i + 1, 0] * size[i + 1]
        strides[dim, 0] = 0
        print("strides:", strides)
        pool = (indices * strides).sum(dim=0)
        i2p = {}
        for i in range(nnz):
            c = int(pool[i])
            if c not in i2p:
                i2p[c] = len(i2p)
            pool[i] = i2p[c]
        print("pool:", pool)
        # compute max
        dense_size = tuple(size[sparse.sparse_dim():])
        mx = torch.empty((pool.max() + 1,) + dense_size, dtype=dtype, device=device)
        mx[:] = -inf
        for n in range(nnz):
            p = pool[n]
            mx[p] = torch.max(mx[p], values[n])
        # apply exp to (v - mx) and sum the results
        exp_values = torch.empty_like(values)
        exp_sums = torch.zeros_like(mx)
        for n in range(nnz):
            p = pool[n]
            v = exp_values[n] = (values[n] - mx[p]).exp()
            exp_sums[p] = exp_sums[p] + v
        # normalize with the sum of exponents
        for n in range(nnz):
            p = pool[n]
            exp_values[n] = exp_values[n] / exp_sums[p]
        return torch.sparse_coo_tensor(indices,
                                       exp_values,
                                       sparse.size(),
                                       dtype=dtype, device=device)
    elif dim < sparse.sparse_dim() + sparse.dense_dim():
        return torch.sparse_coo_tensor(indices,
                                       F.softmax(values, dim - sparse.sparse_dim() + 1),
                                       sparse.size(),
                                       dtype=dtype, device=device)
    else:
        raise ValueError(
            '`dim(=%s)` must be smaller than `sparse_dim(=%s) + dense_dim(=%s)`'
            % (dim, sparse.sparse_dim(), sparse.dense_dim()))


def main():
    a = torch.as_tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(a)
    a_sparse = a.to_sparse()
    print(a_sparse)

    dim = 0

    r0 = a.softmax(dim=dim)
    r1 = sparse_softmax(a_sparse, dim=dim)
    print(r0)
    print(r1.to_dense())


if __name__ == "__main__":
    main()
