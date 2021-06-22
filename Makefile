# GNU Makefile for PFHub BM1

all: spinodal
.PHONY: all spinodal clean

RANKS = 4

spinodal: spinodal.py
	mpirun -np $(RANKS) --mca opal_cuda_support 0 python spinodal.py

clean:
	rm -vf *spinodal.h5 *spinodal.xdmf *spinodal.log
