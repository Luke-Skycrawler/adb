#include "cuda_header.cuh"
using T = float;
__device__ __forceinline__ int maxind(int k, T* S, int n)
{
    int m = k + 1;
    for (int i = k + 2; i < n; ++i)
        if (CUDA_ABS(S[k * n + i]) > CUDA_ABS(S[k * n + m]))
            m = i;
    return m;
}

__device__ __forceinline__ void update(int k, T t, T& y, int& state, T* e, int* change)
{
    y = e[k];
    e[k] = y + t;
    if (change[k] && y == e[k]) {
        change[k] = false;
        --state;
    }
    else if (!change[k] && y != e[k]) {
        change[k] = true;
        ++state;
    }
}

__device__ __forceinline__ void rotate(int k, int l, int i, int j, T s, T c, T* S, int n)
{
    T Skl = S[k * n + l], Sij = S[i * n + j];
    S[k * n + l] = c * Skl - s * Sij;
    S[i * n + j] = s * Skl + c * Sij;
}

__device__ __forceinline__
    T
    makePD(T* S, T* E)
{
    const int n = 12;
    T max_S = 0;
#pragma unroll 12
    for (int i = 0; i < n; ++i)
#pragma unroll 12
        for (int j = 0; j < n; ++j)
            max_S = CUDA_MAX(max_S, CUDA_ABS(S[i * n + j]));
    if (max_S < 1e-12)
        return 0;

    T e[12];
    int k, l, m, state;
    T s, c, t, p, y, d, r;

    int ind[12], changed[12];
#pragma unroll 12
    for (int i = 0; i < n; ++i)
#pragma unroll 12
        for (int j = 0; j < n; ++j)
            E[i * n + j] = (i == j ? 1.0 : 0.0);

    state = n;
    for (k = 0; k < n; ++k) {
        ind[k] = maxind(k, S, n);
        e[k] = S[k * n + k];
        changed[k] = true;
    }

    //
    while (state != 0) {
        m = 0;
        for (k = 1; k < n - 1; ++k) {
            if (CUDA_ABS(S[k * n + ind[k]]) > CUDA_ABS(S[m * n + ind[m]]))
                m = k;
        }
        k = m;
        l = ind[m];
        p = S[k * n + l];
        if (CUDA_ABS(p) < max_S * 1e-9)
            break;
        y = (e[l] - e[k]) / 2.;
        d = CUDA_ABS(y) + CUDA_SQRT(p * p + y * y);
        r = CUDA_SQRT(p * p + d * d);
        c = d / r;
        s = p / r;
        t = p * p / d;
        if (y < 0) {
            s = -s;
            t = -t;
        }
        S[k * n + l] = 0.;
        update(k, -t, y, state, e, changed);
        update(l, t, y, state, e, changed);
        for (int i = 0; i <= k - 1; ++i)
            rotate(i, k, i, l, s, c, S, n);
        for (int i = k + 1; i <= l - 1; ++i)
            rotate(k, i, i, l, s, c, S, n);
        for (int i = l + 1; i < n; ++i)
            rotate(k, i, l, i, s, c, S, n);
#pragma unroll 12
        for (int i = 0; i < n; ++i) {
            T Eik = E[i * n + k], Eil = E[i * n + l];
            E[i * n + k] = c * Eik - s * Eil;
            E[i * n + l] = s * Eik + c * Eil;
        }

#pragma unroll 12
        for (int i = 0; i < n; ++i)
            ind[i] = maxind(i, S, n);
    }

#pragma unroll 12
    for (int i = 0; i < n; ++i)
        e[i] = CUDA_MAX(e[i], 0.);

#pragma unroll 12
    for (int i = 0; i < n; ++i)
#pragma unroll 12
        for (int j = 0; j < n; ++j)
            S[i * n + j] = 0.;

#pragma unroll 12
    for (int i = 0; i < n; ++i)
#pragma unroll 12
        for (int j = 0; j < n; ++j)
#pragma unroll 12
            for (int k = 0; k < n; ++k)
                S[i * n + k] += E[i * n + j] * e[j] * E[k * n + j];
    T max_e = e[0];
#pragma unroll 12
    for (int i = 0; i < n; ++i)
        max_e = CUDA_MAX(max_e, e[i]);
    return max_e;
}
