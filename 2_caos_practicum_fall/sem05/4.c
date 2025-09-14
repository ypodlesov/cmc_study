#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <stdlib.h>

#define deb printf("ok\n")

typedef struct File 
{
    int inode;
    char *path;
} File;

int 
cmp(const void *a, const void *b) 
{
    File f1 = *(File *) a;
    File f2 = *(File *) b;
    if (f1.inode == f2.inode) {
        if (f1.path == NULL) {
            return 0;
        } else {
            return strcmp(f2.path, f1.path);
        }
    }
    return f1.inode < f2.inode ? -1 : 1;
}

int 
cmp2(const void *a, const void *b)
{
    File f1 = *(File *) a;
    File f2 = *(File *) b;
    if (f1.inode == -1) {
        return -1;
    }
    return strcmp(f1.path, f2.path);

}

void 
mycpy(File *f1, File *f2) 
{
    f1->inode = f2->inode;
    f1->path = f2->path;
}

int
unique(File *files, int size) 
{
    int dst = -1, src = 0;
    while (src < size) {
        if (files[src].inode == -1) {
            ++src;
            continue;
        }
        if (dst < 0 || files[src].inode != files[dst].inode) {
            mycpy(&files[++dst], &files[src++]);
        } else {
            ++src;
        }
    }
    return dst + 1;
}


int 
main(int argc, char *argv[])
{
    if (argc < 2) {
        return 0;
    }
    File *files = calloc(argc - 1, sizeof(File));
    for (int i = 1; i < argc; ++i) {
        struct stat cur;
        if (stat(argv[i], &cur) < 0) {
            files[i-1].inode = -1;
            files[i-1].path = NULL;
            continue;
        }
        files[i-1].inode = cur.st_ino;
        files[i-1].path = argv[i];
    }
    qsort(files, argc - 1, sizeof(File), cmp);
    int new_size = unique(files, argc - 1);
    //files = realloc(files, new_size * sizeof(*files));
    qsort(files, new_size, sizeof(File), cmp2);
    for (int i = 0; i < new_size; ++i) {
        printf("%s\n", files[i].path);
    }
    free(files);
    deb;

    return 0;
}
