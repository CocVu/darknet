CUDNN=0
CUDNN_HALF=0
OPENCV=1
AVX=0
OPENMP=1
LIBSO=0

# set GPU=1 and CUDNN=1 to speedup on GPU
# set CUDNN_HALF=1 to further speedup 3 x times (Mixed-precision using Tensor Cores) on GPU Tesla V100, Titan V, DGX-2
# set AVX=1 and OPENMP=1 to speedup on CPU (if error occurs then set AVX=0)

DEBUG=0

ARCH= -gencode arch=compute_30,code=sm_30 \
			-gencode arch=compute_35,code=sm_35 \
			-gencode arch=compute_50,code=[sm_50,compute_50] \
			-gencode arch=compute_52,code=[sm_52,compute_52] \
			-gencode arch=compute_61,code=[sm_61,compute_61]

OS := $(shell uname)

# Tesla V100
# ARCH= -gencode arch=compute_70,code=[sm_70,compute_70]

# GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
# CPU only not
# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61

# GP100/Tesla P100 � DGX-1
# ARCH= -gencode arch=compute_60,code=sm_60

# For Jetson TX1, Tegra X1, DRIVE CX, DRIVE PX - uncomment:
# ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]

# For Jetson Tx2 or Drive-PX2 uncomment:
# ARCH= -gencode arch=compute_62,code=[sm_62,compute_62]


VPATH=./src/
EXEC=darknet
OBJDIR=./obj/

ifeq ($(LIBSO), 1)
LIBNAMESO=darknet.so
APPNAMESO=uselib
endif

CC=gcc
# CC=g++
CPP=g++
NVCC=nvcc
OPTS=-Ofast
LDFLAGS= -lm -pthread
COMMON=
CFLAGS=-Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -g

ifeq ($(DEBUG), 1)
OPTS= -O0 -g
else
ifeq ($(AVX), 1)
CFLAGS+= -ffp-contract=fast -mavx -msse4.1 -msse4a
endif
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv`
COMMON+= `pkg-config --cflags opencv`
endif

ifeq ($(OPENMP), 1)
CFLAGS+= -fopenmp
LDFLAGS+= -lgomp
endif

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
ifeq ($(OS),Darwin) #MAC
LDFLAGS+= -L/usr/local/cuda/lib -lcuda -lcudart -lcublas -lcurand
else
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif
endif

ifeq ($(CUDNN), 1)
COMMON+= -DCUDNN
ifeq ($(OS),Darwin) #MAC
CFLAGS+= -DCUDNN -I/usr/local/cuda/include
LDFLAGS+= -L/usr/local/cuda/lib -lcudnn
else
CFLAGS+= -DCUDNN -I/usr/local/cudnn/include
LDFLAGS+= -L/usr/local/cudnn/lib64 -lcudnn
endif
endif

ifeq ($(CUDNN_HALF), 1)
COMMON+= -DCUDNN_HALF
CFLAGS+= -DCUDNN_HALF
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70]
endif

BASIS_OBJ=http_stream.o gemm.o utils.o cuda.o convolutional_layer.o list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o connected_layer.o cost_layer.o parser.o option_list.o  detection_layer.o captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o coco.o dice.o yolo.o detector.o layer.o compare.o classifier.o local_layer.o swag.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o rnn_vid.o crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o region_layer.o reorg_layer.o reorg_old_layer.o super.o voxel.o tree.o yolo_layer.o upsample_layer.o

ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
BASIS_OBJ+=convolutional_kernels.o activation_kernels.o im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o
endif

DARKNET=$(BASIS_OBJ) darknet.o

DARKNETS = $(addprefix $(OBJDIR), $(DARKNET))

BLOB_OBJ= $(BASIS_OBJ) Blob.o adapter.o detect_blob.o

BLOB_OBJS = $(addprefix $(OBJDIR), $(BLOB_OBJ))

CROP_OBJ= $(BASIS_OBJ) Blob.o blob_and_crop_image.o
CROP_OBJS = $(addprefix $(OBJDIR), $(CROP_OBJ))

DEPS = $(wildcard src/*.h) Makefile

# obj/detect_blob.o: src/detect_blob.cpp
#  g++ -std=c++11   -DOPENCV `pkg-config --cflags opencv` -Wall -Wfatal-errors -Wno-unused-result -Wno-unknown-pragmas -Ofast -DOPENCV -fopenmp -c ./src/detect_blob.cpp -o obj/detect_blob.o

all: obj backup results $(EXEC) $(LIBNAMESO) $(APPNAMESO)

ifeq ($(LIBSO), 1)
CFLAGS+= -fPIC

$(LIBNAMESO): $(DARKNETS) src/yolo_v2_class.hpp src/yolo_v2_class.cpp
	$(CPP) -shared -std=c++11 -fvisibility=hidden -DYOLODLL_EXPORTS $(COMMON) $(CFLAGS) $(DARKNETS) src/yolo_v2_class.cpp -o $@ $(LDFLAGS)

$(APPNAMESO): $(LIBNAMESO) src/yolo_v2_class.hpp src/yolo_console_dll.cpp
	$(CPP) -std=c++11 $(COMMON) $(CFLAGS) -o $@ src/yolo_console_dll.cpp $(LDFLAGS) -L ./ -l:$(LIBNAMESO)

endif

$(EXEC): $(DARKNETS)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

detect_blob: $(BLOB_OBJS)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

crop_blob: $(CROP_OBJS)
	$(CPP) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

run_detect_blob:
	make detect_blob -j4
	./detect_blob

run_blob_crop:
	make crop_blob -j4
	./crop_blob ~/Videos/ngo-may-2.mp4 1000 10000 5

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

# obj/Blob.o: src/Blob.cpp
#		g++ -o obj/Blob.o src/Blob.cpp -c `pkg-config opencv --cflags --libs`

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP)  -std=c++11 $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(DARKNETS) $(EXEC) $(LIBNAMESO) $(APPNAMESO)
