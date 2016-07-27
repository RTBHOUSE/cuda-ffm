SUFFIXES += .d

CUDA_SOURCES  := $(shell find src -name "*.cu")
CXX_SOURCES   := $(shell find src -name "*.cpp")
SOURCES       := $(CUDA_SOURCES) $(CXX_SOURCES)

CUDA_DEPFILES := $(patsubst %.cu,%.d, $(CUDA_SOURCES))
CXX_DEPFILES  := $(patsubst %.cpp,%.d, $(CXX_SOURCES))
DEPFILES      := $(CUDA_DEPFILES) $(CXX_DEPFILES)

CUDA_OBJECTS  := $(patsubst %.cu,%.o, $(CUDA_SOURCES))
CXX_OBJECTS   := $(patsubst %.cpp,%.o, $(CXX_SOURCES))
OBJECTS       := $(CUDA_OBJECTS) $(CXX_OBJECTS)

CUDA_EXECUTABLES := bin/benchmark bin/learn 
CXX_EXECUTABLES  := bin/text_to_bin bin/bin_to_text bin/shuffler bin/splitter bin/predict
EXECUTABLES      := $(CUDA_EXECUTABLES) $(CXX_EXECUTABLES)
TRAINER_OBJS     := src/ffm_trainer.o src/model.o
NODEPS := clean

CUDA_PATH     ?= /usr/local/cuda-7.5
CUB_PATH      := ./cub-1.5.2

HOST_ARCH     := $(shell uname -m)
HOST_OS       := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
HOST_COMPILER ?= g++
TARGET_SIZE   := 64
TARGET_OS     ?= $(HOST_OS)

NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
CXXFLAGS      := -m$(TARGET_SIZE) -std=c++11 -Wall -Wextra -Werror -mavx
NVCC_CCFLAGS  := -m$(TARGET_SIZE) -std=c++11
GENCODE_FLAGS := -gencode arch=compute_52,code=compute_52
INCLUDES      := -isystem $(CUB_PATH) -I./java/src/main/resources/com/rtbhouse/model/natives

ifeq ($(dbg),1)
      NVCC_CCFLAGS += -g -G -O0 -lineinfo
      CXXFLAGS     += -g -ggdb -O0
else
      NVCC_CCFLAGS += -O3 
      CXXFLAGS     += -O3
endif

NVCC_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
NVCC_LDFLAGS :=
NVCC_LDFLAGS += $(NVCC_CCFLAGS)

PURE_CXX_COMPILER := $(HOST_COMPILER) $(CXXFLAGS) $(INCLUDES)

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
	@$(PURE_CXX_COMPILER) -MM -MT '$(patsubst %.cpp,%.o,$<)' $< -MF $@

$(CUDA_DEPFILES): %.d: %.cu $(CUB_PATH)/cub/cub.cuh
	@echo cuda_dep $@
	@$(NVCC) -E -Xcompiler "-isystem /include -isystem $(CUDA_PATH)/include -MM" $(INCLUDES) $(NVCC_CCFLAGS) $< -o $@

$(CXX_OBJECTS): %.o: %.cpp %.d
	@echo pure_cxx $@
	@$(PURE_CXX_COMPILER) -o $@ -c $<

$(CUDA_OBJECTS): %.o: %.cu %.d
	@echo cuda_cxx $@
	@$(NVCC) $(INCLUDES) $(NVCC_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

bin/benchmark: src/benchmark.o $(TRAINER_OBJS)
	@echo cuda_link $@
	@$(NVCC) $(NVCC_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+

bin/learn: src/learn.o src/learn_options.o $(TRAINER_OBJS)
	@echo cuda_link $@
	@$(NVCC) $(NVCC_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+
	
$(CXX_EXECUTABLES): bin/% : src/%.o src/model.o
	@echo cxx_link $@
	@$(HOST_COMPILER) ${CXXFLAGS} -o $@ $+


ifeq (0, $(words $(findstring $(MAKECMDGOALS), $(NODEPS)))) # ignore dep. generation when cleaning
-include $(DEPFILES)
endif
