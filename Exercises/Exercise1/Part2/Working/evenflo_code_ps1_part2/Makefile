CC=gcc
CFLAGS=-std=c99 -Wall -Wextra -Wpedantic

all : mandel_serial mandel_mpi
	

mandel_serial: mandel_serial.c
	gcc -o mandel_serial mandel_serial.c $(CFLAGS)
	
mandel_mpi: mandel_mpi.c
	mpicc -o mandel_mpi mandel_mpi.c $(CFLAGS)

.PHONY: run clean


clean:
	rm -f mandel_serial mandel_mpi mandel2.bmp
