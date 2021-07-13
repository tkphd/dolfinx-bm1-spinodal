# GNU Makefile for PFHub BM1
# Recommended for use with a Conda environment for Singularity with Python 3

# Cluster Settings

MPI = mpirun
PY3 = python3
RANKS = 4

# Container Settings

IMAGE = dolfinx/dolfinx
NAME = fenicsx

# Make Targets

all: dolfinx-bm-1b.xdmf
.PHONY: all clean instance list shell spinodal stop watch

dolfinx-bm-1b.xdmf: spinodal.py
	make instance
	make spinodal
	make stop

clean:
	rm -vf *spinodal.h5 *spinodal.xdmf *spinodal.log dolfinx*.csv

instance:
	singularity instance start -H $(PWD) docker://$(IMAGE) $(NAME)

list:
	singularity instance list

shell:
	singularity exec instance://$(NAME) bash --init-file .singular-prompt

spinodal: spinodal.py
	singularity exec instance://$(NAME) $(MPI) -np $(RANKS) $(PY3) -u spinodal.py

stop:
	singularity instance stop $(NAME)

watch:
	singularity exec instance://$(NAME) bash -c "watch cat dolfinx-bm-1b.csv"

#docker:
#	docker run --name $(NAME) --rm -ti -v $(PWD):/root/shared -w /root/shared $(IMAGE) bash --init-file .singular-prompt
