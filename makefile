CUDA_INSTALL_PATH = /program/cuda
GCC_INSTALL_PATH = /program/icc/composer_xe_2013.0.079

NVCC = $(CUDA_INSTALL_PATH)/bin/nvcc
GCC = $(GCC_INSTALL_PATH)/bin/intel64/icc

LDFLAGS = -L$(CUDA_INSTALL_PATH)/lib64

LIB1 = -lcudart -lcurand -lcufft
LIB2 = -lstdc++

CFILES1 = LSRTM_PP_ISO_disp2d.cpp
CFILES2 = LSRTM_PP_ISO_disp2d_function.cpp
CUFILE1 = LSRTM_PP_ISO_disp2d_forward.cu
CUFILE7 = LSRTM_PP_ISO_disp2d_kernel.cu
OBJECTS = LSRTM_PP_ISO_disp2d.o LSRTM_PP_ISO_disp2d_function.o LSRTM_PP_ISO_disp2d_forward.o LSRTM_PP_ISO_disp2d_kernel.o

EXECNAME = afm_pp_iso_q_ncq_fd_forder

all:
	$(GCC) -c $(CFILES1)	
	$(GCC) -c $(CFILES2)
	$(NVCC) -c $(CUFILE1)
	$(NVCC) -c $(CUFILE7)
	$(GCC) -o $(EXECNAME) $(OBJECTS) $(LIB1) $(LDFLAGS) $(LIB2)

clean:
	rm -f *.o
