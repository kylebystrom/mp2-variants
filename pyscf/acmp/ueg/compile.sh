export COMPILE_FLAGS="-shared -Wall -g -O3 -fopenmp -pthread -fPIC"
gcc $COMPILE_FLAGS -o libmp2.so mp2.c
gcc $COMPILE_FLAGS -o libacfq.so acfq.c

