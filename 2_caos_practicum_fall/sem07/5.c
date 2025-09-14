#include <stdio.h>
#include <time.h>
#include <sys/time.h>

enum 
{
    PERIOD = 29 * 24 * 60 * 60 + 12 * 60 * 60 + 44 * 60,
    SEC = 24 * 60 * 60,
    D_SEC = 0,
    D_MIN = 14,
    D_HOUR = 11,
    D_MDAY = 26,
    D_MON = 4,
    D_YEAR = 2021,
    DAY = 256,
    START_YEAR = 1900,
    MONDAY = 1
};

//const struct tm DAY = {
//    tm_sec = 0,
//    tm_min = 14,
//    tm_hour = 11,
//    tm_mday = 26,
//    tm_mon = 4,
//    tm_year = 121,
//};

int
main(int argc, char *argv[])
{

    struct tm date = {};
    date.tm_sec = D_SEC;
    date.tm_min = D_MIN;
    date.tm_hour = D_HOUR;
    date.tm_mday = D_MDAY;
    date.tm_mon = D_MON;
    date.tm_year = D_YEAR - START_YEAR;
    date.tm_isdst = -1;
    timegm(&date);

    int year;
    scanf("%d", &year);
    while (date.tm_year >= year - START_YEAR) {
        date.tm_sec -= PERIOD;
        timegm(&date);
    }
    while (date.tm_year < year - START_YEAR) {
        date.tm_sec += PERIOD;
        timegm(&date);
    }
    while (date.tm_year == year - START_YEAR && date.tm_yday < DAY) {
        date.tm_sec += PERIOD;
        timegm(&date);
    }
    date.tm_sec += SEC;
    timegm(&date);
    while (date.tm_wday != MONDAY) {
        date.tm_sec += SEC;
        timegm(&date);
    }
    for (int i = 1; i < 4; ++i) {
        date.tm_sec += 7 * SEC;
        timegm(&date);
    }
    printf("%d-%02d-%02d\n", date.tm_year + START_YEAR, date.tm_mon + 1, date.tm_mday);


    
    return 0;
}
