SUFFIXES += .d

CUDA_FFM_VERSION := "cuda-ffm-v0.2.9"

CUDA_SOURCES  := $(shell find src -name "*.cu")
CXX_SOURCES   := $(shell find src -name "*.cpp")
SOURCES       := $(CUDA_SOURCES) $(CXX_SOURCES)

CUDA_DEPFILES := $(patsubst %.cu,%.d, $(CUDA_SOURCES))
CXX_DEPFILES  := $(patsubst %.cpp,%.d, $(CXX_SOURCES))
DEPFILES      := $(CUDA_DEPFILES) $(CXX_DEPFILES)

CUDA_OBJECTS  := $(patsubst %.cu,%.o, $(CUDA_SOURCES))
CXX_OBJECTS   := $(patsubst %.cpp,%.o, $(CXX_SOURCES))
OBJECTS       := $(CUDA_OBJECTS) $(CXX_OBJECTS)

CUDA_EXECUTABLES := bin/benchmark bin/trainer
CXX_EXECUTABLES  := bin/text_to_bin bin/bin_to_text bin/shuffler bin/splitter bin/predict
EXECUTABLES      := $(CUDA_EXECUTABLES) $(CXX_EXECUTABLES)
TRAINER_OBJS     := src/ffm_trainer.o src/ffm_predictor.o src/model.o src/cuda_utils.o
NODEPS := clean

CUDA_PATH     ?= /usr/local/cuda-7.5
CUB_PATH      := ./cub-1.5.2

HOST_ARCH     := $(shell uname -m)
HOST_OS       := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_SIZE   := 64
TARGET_OS     ?= $(HOST_OS)

HOST_COMPILER        ?= g++ # nvcc does not work with gcc-6
DECENT_HOST_COMPILER ?= g++-6 # gcc-4.8+ is OK, but gcc-6 produces faster code
NVCC                 := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
CXXFLAGS_CPU         ?= -mfma -mavx2 # -mavx is minimum
CXXFLAGS             := -m$(TARGET_SIZE) -std=c++11 -Wall -Wextra -Werror -DCUDA_FFM_VERSION=\"$(CUDA_FFM_VERSION)\" $(CXXFLAGS_CPU)

NVCC_CCFLAGS  := -m$(TARGET_SIZE) -std=c++11
GENCODE_FLAGS ?= -gencode arch=compute_52,code=compute_52
INCLUDES      := -isystem $(CUB_PATH) -I./java/src/main/resources/com/rtbhouse/model/natives

ifeq ($(dbg),1)
      NVCC_CCFLAGS += -g -G -O0 -lineinfo
      OPT_CXXFLAGS += -g -ggdb -O0
else
      NVCC_CCFLAGS += -O3
      OPT_CXXFLAGS += -O3 -funroll-all-loops -funroll-loops -floop-nest-optimize -ffast-math
endif

CXXFLAGS     += $(OPT_CXXFLAGS)
NVCC_CCFLAGS += $(addprefix -Xcompiler ,$(OPT_CXXFLAGS))
NVCC_LDFLAGS :=
NVCC_LDFLAGS += $(NVCC_CCFLAGS)

PURE_CXX_COMPILE := $(DECENT_HOST_COMPILER) $(CXXFLAGS) $(INCLUDES)

all: build

build: bin $(EXECUTABLES)

bin:
	@mkdir bin

clean:
	rm -rf src/*.o src/*.d bin

$(CUB_PATH)/cub/cub.cuh:
	wget https://github.com/NVlabs/cub/archive/1.5.2.tar.gz -O - | tar zxf -

$(CXX_DEPFILES): %.d: %.cpp $(CUB_PATH)/cub/cub.cuh
	@echo cxx_dep $@
	@$(PURE_CXX_COMPILE) -MM -MT '$(patsubst %.cpp,%.o,$<)' $< -MF $@

$(CUDA_DEPFILES): %.d: %.cu $(CUB_PATH)/cub/cub.cuh
	@echo cuda_dep $@
	@$(NVCC) -E -Xcompiler "-isystem /include -isystem $(CUDA_PATH)/include -MM" $(INCLUDES) $(NVCC_CCFLAGS) $< -o $@

$(CXX_OBJECTS): %.o: %.cpp %.d
	@echo pure_cxx $@
	@$(PURE_CXX_COMPILE) -o $@ -c $<

$(CUDA_OBJECTS): %.o: %.cu %.d
	@echo cuda_cxx $@
	@$(NVCC) $(INCLUDES) $(NVCC_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bin/benchmark: src/benchmark.o $(TRAINER_OBJS)
	@echo cuda_link $@
	@$(NVCC) $(NVCC_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

bin/trainer: src/trainer.o src/training_options.o src/training_history.o src/training_session.o $(TRAINER_OBJS)
	@echo cuda_link $@
	@$(NVCC) $(NVCC_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

$(CXX_EXECUTABLES): bin/% : src/%.o src/model.o
	@echo cxx_link $@
	@$(DECENT_HOST_COMPILER) ${CXXFLAGS} -o $@ $+


ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS)))) # ignore dep. generation when cleaning
-include $(DEPFILES)
endif
