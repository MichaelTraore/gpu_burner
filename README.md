# gpu-burn
Single-GPU CUDA stress test based on thrust. This is a fork from
http://wili.cc/blog/gpu-burn.html

# Easy docker build and run

```
git clone https://github.com/MichaelTraore/gpu_burner.git
cd gpu-burner
docker build -t gpu_burner .
docker run --rm --gpus all gpu_burner
```

# Building
To build GPU Burn:
```
docker build -t gpu_burner .
```
Or directly with cmake :

```
cmake build <your_build_directory>
```


# Usage

    GPU Burn
    Usage: docker run --rm --gpus all gpu_burner [TIME_IN_SECONDS]
    Example:
    gpu_burner 3600

    The default TIME_IN_SECONDS is 30 
