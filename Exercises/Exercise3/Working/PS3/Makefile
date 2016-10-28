CFLAGS+=-std=c99 -O3
LDLIBS=-lm -pthread -fopenmp
TARGETS=heat_serial heat_omp heat_pthread

all: ${TARGETS}


clean:
	-rm -f ${TARGETS}
	-rm -f data/*
