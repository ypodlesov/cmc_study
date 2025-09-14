CC = gcc 
CFLAGS = -Wall -Werror -O2 -fsanitize=undefined -fsanitize=address
LDFLAGS = -s
TARGET = main

CFILES = 1.c
OBJECTS = ${CFILES:.c=.o}

all : ${TARGET}

${TARGET} : ${CFILES}
		${CC} ${CFLAGS} $^ -o ${TARGET}

clean : 
		rm -f ${TARGET} ${OBJECTS}
