
int 
is_duodots(char *str) 
{
    if (str[1] == '\0') return 0;
    return (str[0] == '.' && str[1] == '.' && (str[2] == '\0' || str[2] == '/'));
}

int 
is_onedot(char *str) 
{
    return (str[0] == '.' && (str[1] == '\0' || str[1] == '/'));
}

void 
normalize_path(char *buf) 
{
    int dst = 0;
    int src = 1;
    while (buf[src] != '\0') {
        if (is_duodots(&buf[src])) {
            src += buf[src + 2] ? 3 : 2;
            if (dst > 0) {
                --dst;
                while (dst > 0 && buf[dst] != '/') {
                    --dst;
                }
            }
        } else if (is_onedot(&buf[src])) {
            src += buf[src + 1] == '/' ? 2 : 1;
        } else {
            while (buf[src] != '\0' && buf[src] != '/') {
                buf[++dst] = buf[src++];
            }
            if (buf[src] == '/') {
                buf[++dst] = buf[src++];
            }
        }
    }
    if (dst > 0 && buf[dst] == '/') {
        --dst;
    }
    for (int i = dst + 1; i < src; ++i) {
        buf[i] = '\0';
    }
}
