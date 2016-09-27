rootlibs=$(shell root-config --libs)
rootflags=$(shell root-config --cflags)
otherflags=-I. -L/usr/lib64/nvidia -L$(CUDASYS)/lib64 -I$(CUDASYS)/include/ -L$(TBB_LIB)

all: fitPulses

pulsefitting_kernel.o: pulsefitting_kernel.cu header.h
	nvcc -c pulsefitting_kernel.cu -std=c++11 -O3

fitPulses: fitPulses.cxx pulsefitting_kernel.o header.h
	g++ $^ -o $@ -O3 -std=c++11 -Wall $(rootflags) $(rootlibs) $(otherflags) -ltbb -lcuda -lcudart
