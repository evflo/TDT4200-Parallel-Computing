CFLAGS = -std=c99 -O3 -mavx2 -march=native -mno-avx
LDLIBS = -lcblas -latlas -lm 
all: chemm_naive chemm_atlas chemm_fast

chemm_naive : chemm.c naive.o
	gcc $(CFLAGS) $(LDFLAGS) chemm.c naive.o -o chemm_naive -framework Accelerate -L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers

chemm_atlas : chemm.c atlas.o
	gcc $(CFLAGS) $(LDFLAGS) chemm.c atlas.o -o chemm_atlas  -framework Accelerate -L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers

chemm_fast : chemm.c fast.o
	gcc $(CFLAGS) $(LDFLAGS) chemm.c fast.o -o chemm_fast -framework Accelerate -L/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A -I/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers

%.o : %.c
	gcc $(CFLAGS) -O3 -c $< $(LDLIBS)

clean:
	rm -f chemm_* *.o
