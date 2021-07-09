# GNU Makefile for PFHub BM1

RANKS = 4

all: spinodal
.PHONY: all docker-launch docker-run spinodal updock watcher clean

docked: spinodal.py
	docker run --name fenicsx --rm -ti -v $(PWD):/root/shared -w /root/shared dolfinx/dolfinx bash --init-file .docker-prompt

spinodal: spinodal.py
	$(MAKE) clean
	mpirun -np $(RANKS) python3 -u spinodal.py

#spinodal: spinodal.py
#	mpirun -np $(RANKS) --mca opal_cuda_support 0 python -u spinodal.py

updock:
	docker pull dolfinx/dolfinx

watcher:
	docker exec -t fenicsx bash -c "watch cat dolfinx-bm-1b.csv; exit"

clean:
	rm -vf *spinodal.h5 *spinodal.xdmf *spinodal.log dolfinx*.csv
