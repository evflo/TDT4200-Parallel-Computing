CC=gcc
CFLAGS=-std=c99 -Wall -Wextra -Wpedantic -O0 # No compiler optimization

complex: complex.c
	$(CC) -o $@ $< $(CFLAGS)

.PHONY: run clean

run: complex
	./$<

clean:
	rm -f complex
