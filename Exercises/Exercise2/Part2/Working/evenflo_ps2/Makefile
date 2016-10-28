CC=mpicc
CFLAGS+=-std=c99 -O3
LDLIBS=-lm
TARGETS=heat
NP=8
all: ${TARGETS}

run: ${TARGETS}
	mpirun -np ${NP} heat

clean:
	-rm -f ${TARGETS}
	-rm -f data/*
