/*
    https://rosettacode.org/wiki/Farey_sequence#C
*/

#ifdef DEBUG
    #include <stdio.h>
#endif

#include <stdlib.h>
#include <string.h>
typedef unsigned long long ull;

void farey(int n, ull len, int* ndarray)
{
    typedef struct { int d, n; } frac;
    frac f1 = {0, 1}, f2 = {1, n}, t;
    int k;
    
    ndarray[0] = 0;
    ndarray[1] = 1;
    ndarray[2] = 1;
    ndarray[3] = n;

    int i = 4;
    while (f2.n > 1) {
        k = (n + f1.n) / f2.n;
        t = f1, f1 = f2, f2 = (frac) { f2.d * k - t.d, f2.n * k - t.n };
        ndarray[i++] = f2.d;
        ndarray[i++] = f2.n;
    }
}

ull *cache;
size_t ccap;

ull farey_len(int n)
{
    if (n >= ccap) {
        size_t old = ccap;
        if (!ccap) ccap = 16;
        while (ccap <= n) ccap *= 2;
        cache = realloc(cache, sizeof(ull) * ccap);
        memset(cache + old, 0, sizeof(ull) * (ccap - old));
    } else if (cache[n])
        return cache[n];

    ull len = (ull)n*(n + 3) / 2;
    int p, q = 0;
    for (p = 2; p <= n; p = q) {
        q = n/(n/p) + 1;
        len -= farey_len(n/p) * (q - p);
    }

    cache[n] = len;
    return len;
}
