
int 
satsum(int v1, int v2) 
{
    enum { MY_INT_MAX = (1U << 31) - 1};
    enum { MY_INT_MIN = 1U << 31};
    if (v1 < 0 && v2 < 0) {
        int M = -(MY_INT_MIN >> 1);
        if ((v1 < M && v2 < M) || (v1 < M && v2 == M)  || (v1 == M && v2 < M)) return MY_INT_MIN;
        if (v1 < M && v2 + (v1 - M) < M) return MY_INT_MIN;
        if (v2 < M && v1 + (v2 - M) < M) return MY_INT_MIN;
    }
    if (v1 > 0 && v2 > 0) {
        int M = MY_INT_MAX >> 1;
        if (v1 > M && v2 > M) return MY_INT_MAX;
        if (v1 > M && v2 + (v1 - M) > M + 1) return MY_INT_MAX;
        if (v2 > M && v1 + (v2 - M) > M + 1) return MY_INT_MAX;
    }
    return v1 + v2;

    return 0;
}

