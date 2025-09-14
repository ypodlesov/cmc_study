#include "3.cpp"

C func1(const C &v1, int v2)
{
    return C(v2 + v1, ~v1);
}

void func2(const C *p1, double p2)
{
    C v1 = p2;
    C v2[2][2];
    C v3 = func1(func1(func1(&p1[3], p2), ~p1[2]), ++v1 * v2[1]);
}    
   