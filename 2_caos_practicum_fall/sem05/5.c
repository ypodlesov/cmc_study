#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <libgen.h>

enum 
{
    DATA_PORTION = 1 << 12
};

int 
copy_file(const char *srcpath, const char *dstpath) 
{
    if (srcpath == NULL || dstpath == NULL) {
        return -1;
    }
    if (strcmp(srcpath, dstpath) == 0) {
        return 0;
    }
    int src_fd = open(srcpath, O_RDONLY);
    if (access(srcpath, R_OK) || src_fd < 0) {
        return -1;
    }
    struct stat src_stat;
    if (stat(srcpath, &src_stat) == 0 && S_ISREG(src_stat.st_mode)) { 
        char *dst_path = strdup(dstpath);
        struct stat dst_stat;
        stat(dstpath, &dst_stat);
        if (S_ISDIR(dst_stat.st_mode)) {
            char *src_path = strdup(srcpath);
            char *file_name = basename(src_path);
            strcat(dst_path, "/");
            strcat(dst_path, file_name);
        }
        stat(dst_path, &dst_stat);
        if (src_stat.st_ino == dst_stat.st_ino) {
            return close(src_fd);
        }
        int dst_fd = open(dst_path, O_CREAT | O_TRUNC | O_WRONLY, src_stat.st_mode);
        if (dst_fd < 0) {
            return -1;
        }
        char *buf = calloc(DATA_PORTION, sizeof(*buf));
        ssize_t rd_size;
        while ((rd_size = read(src_fd, buf, DATA_PORTION * sizeof(*buf))) > 0) {
            ssize_t wr_size = 0;
            while (rd_size > wr_size) {
                if ((wr_size += write(dst_fd, buf + wr_size, rd_size - wr_size)) < 0) {
                    return -1;
                }
            }
        }
        free(buf); 
        if (rd_size < 0) {
            return -1;
        }
        return close(src_fd) == 0 && close(dst_fd) == 0 ? 0 : -1;
    } 
    return -1;
}
