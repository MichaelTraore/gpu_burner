cmake_minimum_required(VERSION 3.8)
project(GpuBurn)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set source files
set(SOURCES gpu_burn-drv.cpp)
# Set the language for your .cpp file to CUDA
set_source_files_properties(gpu_burn-drv.cpp PROPERTIES LANGUAGE CUDA)

# Specify the CUDA compilation flags for your .cpp file
set_property(SOURCE gpu_burn-drv.cpp PROPERTY CUDA_SOURCE_PROPERTY_FORMAT OBJ)

# Set CUDA-specific flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "61-real;75-real;86-real" CACHE STRING "Target GPU architecture for NVCC")
endif()

# Compile CUDA code
cuda_add_executable(gpu_burner ${SOURCES})
target_link_libraries(gpu_burner
  cuda
)
