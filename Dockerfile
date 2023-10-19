ARG CUDA_VERSION=11.8.0
ARG IMAGE_DISTRO=ubi8

FROM nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS builder

# Use an official CUDA base image as a starting point
# FROM nvidia/cuda:11.8.0-devel-ubi8

# Install necessary packages and dependencies
RUN yum -y update && yum -y install \
    cmake \
    make \
    gcc-c++ \
    wget \
    libX11-devel \
    && yum clean all


# Set the working directory inside the container
WORKDIR /app

# Copy the source code and CMakeLists.txt into the container
COPY . /app/

# Create a build directory and set it as the working directory
RUN mkdir build
WORKDIR /app/build

# Generate Makefiles with CMake and build the project
RUN cmake ..
RUN cmake --build .

RUN chmod +x /app/build/gpu_burner


# Set the entry point to run the built executable
ENTRYPOINT ["/app/build/gpu_burner"]
