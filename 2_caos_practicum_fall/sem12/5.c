#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

enum
{
    MODE = 0700,
    MIN_LEN = 5,
    MOD_LEN = 10,
    MAX_SIZE = 1 << 20
};


void
gen_name(char *res, int size, char *suffix) 
{
    for (int i = 0; i < size; ++i) {
        res[i] = 'a' + rand() % 26;
    }
    if (suffix) strcat(res, suffix);
}

char *
base_name(void)
{
    if (getenv("XDG_RUNTIME_DIR")) {
        return getenv("XDG_RUNTIME_DIR");
    }
    if (getenv("TMPDIR")) {
        return getenv("TMPDIR");
    }
    return strdup("/tmp");
}

int 
fill_script(FILE *file, char *cpath, char *exfile) 
{
    if (!file) return 1;
    char content[MAX_SIZE] = "#! /bin/bash\n";
    strcat(content, "gcc ");
    strcat(content, cpath);
    strcat(content, " -o ");
    strcat(content, exfile);
    strcat(content, "\nrm ");
    strcat(content, cpath);
    strcat(content, "\n./");
    strcat(content, exfile);
    strcat(content, "\nrm ");
    strcat(content, exfile);
    strcat(content, "\nrm $0\n");
    if (fprintf(file, "%s", content) < 0) return 1;
    return 0;
}

int 
fill_cfile(FILE *file, char *expr)  
{
    char content[MAX_SIZE] = "#include <stdio.h>\n";
    strcat(content, "#include <string.h>\n");
    strcat(content, "#define reject \"reject\\n\"\n");
    strcat(content, "#define disqualify \"disqualify\\n\"\n");
    strcat(content, "#define summon \"summon\\n\"\n");
    strcat(content, "void foo(int x) {\n");
    strcat(content, "printf(");
    strcat(content, expr);
    strcat(content, ");}\n");
    strcat(content, "int main() {\n");
    if (fprintf(file, "%s\n", content) < 0) return 1;
    int n;
    while (scanf("%d", &n) == 1) {
        if (fprintf(file, "foo(%d);", n) < 0) return 1;
    }   
    if (fprintf(file, "}\n") < 0) return 1;
    return 0;
}

int
main(int argc, char *argv[]) 
{
    char script_path[PATH_MAX];
    char c_path[PATH_MAX];
    char base[PATH_MAX];
    strcpy(base, base_name());
    strcat(base, "/");
    strcpy(script_path, base);
    strcpy(c_path, base);
    srand(time(NULL));
    char script_name[MOD_LEN << 2] = {};
    gen_name(script_name, MIN_LEN + rand() % MOD_LEN, NULL);
    char c_name[MOD_LEN << 2] = {};
    gen_name(c_name, MIN_LEN + rand() % MOD_LEN, ".c");
    strcat(script_path, script_name);
    strcat(c_path, c_name);
    int c_fd = open(c_path, O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, MODE);
    FILE *c_file = fdopen(c_fd, "w");
    if (fill_cfile(c_file, argv[1])) {
        fprintf(stderr, "error! cannot fill source file\n");
        exit(1);
    }
    fclose(c_file);
    int script_fd = open(script_path, O_CREAT | O_RDWR | O_TRUNC | O_CLOEXEC, MODE);
    FILE *script_file = fdopen(script_fd, "w");
    char exe_name[MOD_LEN << 2] = {};
    gen_name(exe_name, MIN_LEN + rand() % MOD_LEN, NULL);
    if (fill_script(script_file, c_path, exe_name)) {
        fprintf(stderr, "error! cannot write in scriptfile\n");
        exit(1);
    }
    fclose(script_file);
    execlp(script_path, script_path, NULL);
    if (errno == EACCES) printf("ok\n");
    printf("ok\n");
    exit(1);
}
