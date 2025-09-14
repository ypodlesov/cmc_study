#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

enum 
{
    DAY_SECONDS = 24 * 60 * 60
};

int check(time_t val) {
    for (int i = 0; i < 9; ++i) {
        if (val == (1 << i)) {
            return 1;
        }
    }
    return 0;
}

void 
make_null(struct tm *a) 
{
    a->tm_sec = 0;
    a->tm_min = 0;
    a->tm_hour = 0;
    a->tm_mday = 0;
    a->tm_mon = 0;
    a->tm_year = 0;
    a->tm_wday = 0;
    a->tm_yday = 0;
    a->tm_isdst = 0;
}

int
main()
{
    int year;
    scanf("%d", &year);
    struct tm cur;
    make_null(&cur);
    cur.tm_year = year - 1900;
    cur.tm_mday = 1;
    time_t st_time = timegm(&cur); 
    time_t time = timegm(&cur);
    int cnt = 0;
    while (cur.tm_year == year - 1900) {
        time += DAY_SECONDS;
        gmtime_r(&time, &cur);
        if (cur.tm_wday == 0 || cur.tm_wday == 6 || check((time - st_time) / DAY_SECONDS + 1)) {
            ++cnt;
        }
    }
    printf("%d\n", cnt);

    return 0;
}
