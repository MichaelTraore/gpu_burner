/*
 * Copyright (c) 2022, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *	this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 *FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 *DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 *SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 *OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 *those of the authors and should not be interpreted as representing official
 *policies, either expressed or implied, of the FreeBSD Project.
 */

#include <chrono>
#include <cstdio>
#include <thread>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>


// Round a / b to nearest higher integer value
inline std::uint32_t iDivUp(std::uint32_t a, std::uint32_t b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}
struct is_even
{
  __host__ __device__ bool operator()(const double x) { return (static_cast<int>(x) % 2) == 0; }
};

__global__ void compare(double* A, double* faulty_elems, size_t N)
{
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N)
    return;
  auto my_faulty = 0u;
  auto not_valid = is_even{};
  if (not_valid(A[id]))
    my_faulty++;
  faulty_elems[id] = static_cast<double>(my_faulty);
}

void _checkError(cudaError_t rCode, std::string file, int line, std::string desc = "")
{
  if (rCode != CUDA_SUCCESS) {

    const char* err = cudaGetErrorString(rCode);

    throw std::runtime_error(
        (desc == "" ? std::string("Error (") : (std::string("Error in ") + desc + " (")) + file +
        ":" + std::to_string(line) + "): " + err);
    // Yes, this *is* a memory leak, but this block is only executed on
    // error, so it's not a big deal
  }
}

#define checkError(rCode, ...) _checkError(rCode, __FILE__, __LINE__, ##__VA_ARGS__)

void printMemoryInfo(size_t total_memory, size_t free_memory, size_t buffer_size){
  auto megabyte = size_t{1024ul * 1024ul};
  auto device_id = int{};
  auto properties = cudaDeviceProp{};
  checkError(cudaGetDevice(&device_id));
  checkError(cudaGetDeviceProperties(&properties, device_id));
  std::cout << "Initialized device " << properties.name << " (id = " << device_id << ")"
            << " with " << total_memory / megabyte
            << "MB of memory (" << free_memory / megabyte <<"MB available, "
            << "using 2 buffers of " << sizeof(double) * buffer_size / megabyte << " MB each) \n";
}

void printCurrentResult(int elapsed_time, int total_time_percentage, int errors_count){
  std::cout << "\rElapsed time: " << elapsed_time << " seconds";
  std::cout << " (" << total_time_percentage << "%) -- with ";
  std::cout << errors_count << " error(s)";
  std::cout.flush(); // Flush the output
}

int main(int argc, char** argv){


  //Get Memory info
  constexpr auto free_memory_percentage = 0.9f;
  auto free_memory = size_t{0ul};
  auto total_memory = size_t{0ul};
  checkError(cudaMemGetInfo(&free_memory, &total_memory));

  const auto max_size = static_cast<size_t>(free_memory * free_memory_percentage/ sizeof(double));
  // using 2 third of the available memory (2 buffers of buffer_size) 
  const auto buffer_size = max_size / 3ul;
  printMemoryInfo(total_memory, free_memory, buffer_size);
  // init buffers
  auto d_A = thrust::device_vector<double>{buffer_size};
  auto d_B = thrust::device_vector<double>{buffer_size};

  // read the desired duration in seconds (30 seconds by default)
  const auto desiredDuration = (argc == 2)? std::atoi(argv[1]) : 30; 

  // Calculate the end time
  const auto start_time = std::chrono::high_resolution_clock::now();
  auto last_print_time = start_time;
  const auto end_time = std::chrono::high_resolution_clock::now() + std::chrono::seconds(desiredDuration);

  auto errors_count = 0u;

  // Your loop
  while (std::chrono::high_resolution_clock::now() < end_time) {
      // Your loop logic goes here

    thrust::sequence(d_A.begin(), d_A.end());
    thrust::sequence(d_B.begin(), d_B.end());
    thrust::sort(d_A.begin(), d_A.end(), thrust::greater<double>());
    thrust::sort(d_A.begin(), d_A.end(), thrust::less<double>());
    auto new_last = thrust::remove_if(d_A.begin(), d_A.end(), d_B.begin(), is_even());
    auto new_size = thrust::distance(d_A.begin(), new_last);
    constexpr auto g_blockSize = 16u;
    compare<<<iDivUp(new_size, g_blockSize), g_blockSize>>>(
      thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()), new_size);

    checkError(cudaDeviceSynchronize());

    errors_count += static_cast<int>(thrust::reduce(d_B.begin(), d_B.begin() + new_size));
    const auto now = std::chrono::high_resolution_clock::now();
    // update log each second
    if(now >= last_print_time +  std::chrono::seconds(1)){
      const auto elapsed_time = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
      double total_time_percentage = static_cast<int>(elapsed_time * 100.0 / desiredDuration );
      printCurrentResult(elapsed_time, total_time_percentage, errors_count);
      last_print_time = now;
    }

  }
  printCurrentResult(std::chrono::seconds(desiredDuration).count(), 100, errors_count);

  std::cout << "\n Test ended with " << errors_count << " error(s) ! \n";
  return 0;
}
