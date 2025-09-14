#include <time.h>
#include <sys/time.h>
#include <stdio.h>

enum 
{
    WEEK_SECONDS = 7 * 24 * 60 * 60,
    DAY_SECONDS = 24 * 60 * 60
};

int 
main() 
{
    long year;
    scanf("%ld", &year);
    struct tm cur;
    cur.tm_sec = 0;
    cur.tm_min = 0;
    cur.tm_hour = 0;
    cur.tm_mday = 1;
    cur.tm_mon = 0;
    cur.tm_year = year - 1900;
    time_t time = timegm(&cur);
    while (cur.tm_year == year - 1900 && cur.tm_wday != 4) {
        time += DAY_SECONDS;
        gmtime_r(&time, &cur);
    }

    while (cur.tm_year == year - 1900) {
        if (cur.tm_wday == 4 && cur.tm_mday % 3 && ((cur.tm_mday + 6) / 7 == 2 || (cur.tm_mday + 6) / 7 == 4)) {
            printf("%d %d\n", cur.tm_mon + 1, cur.tm_mday);
        }
        time += WEEK_SECONDS;
        gmtime_r(&time, &cur);
    }


    return 0;
}
