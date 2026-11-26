#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
constexpr int kThreadsPerBlock = 256;

struct AppConfig {
  std::string input_path = "data/input/sample_00.ppm";
  std::string output_path = "data/output/sample_00_gray.pgm";
};

void CheckCuda(cudaError_t err, const char* func, const char* file, int line) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s:%d (%s) in %s: %s\n",
                 file, line, cudaGetErrorName(err), func,
                 cudaGetErrorString(err));
    std::exit(EXIT_FAILURE);
  }
}

#define CHECK_CUDA(val) CheckCuda((val), #val, __FILE__, __LINE__)

AppConfig ParseArgs(int argc, char** argv) {
  AppConfig config;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
      config.input_path = argv[++i];
    } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      config.output_path = argv[++i];
    }
  }
  return config;
}
}  // namespace

__global__ void RgbToGrayscaleKernel(const unsigned char* __restrict__ rgb,
                                   unsigned char* __restrict__ gray,
                                   int width, int height) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_pixels = width * height;
  if (idx < total_pixels) {
    int rgb_idx = idx * 3;
    unsigned char r = rgb[rgb_idx];
    unsigned char g = rgb[rgb_idx + 1];
    unsigned char b = rgb[rgb_idx + 2];
    float lum = 0.299f * static_cast<float>(r) +
                0.587f * static_cast<float>(g) +
                0.114f * static_cast<float>(b);
    gray[idx] = static_cast<unsigned char>(lum);
  }
}

bool LoadPpm(const std::string& path, int* width, int* height,
             std::vector<unsigned char>* pixels) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) return false;
  std::string magic;
  int max_val = 255;
  in >> magic >> *width >> *height >> max_val;
  in.get();
  if (magic != "P6" || *width <= 0 || *height <= 0) return false;
  pixels->resize(static_cast<size_t>(*width) * (*height) * 3);
  in.read(reinterpret_cast<char*>(pixels->data()),
          static_cast<std::streamsize>(pixels->size()));
  return in.good();
}

bool SavePgm(const std::string& path, int width, int height,
             const std::vector<unsigned char>& gray) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) return false;
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(gray.data()),
            static_cast<std::streamsize>(gray.size()));
  return out.good();
}

int main(int argc, char** argv) {
  AppConfig config = ParseArgs(argc, argv);
  int width = 0;
  int height = 0;
  std::vector<unsigned char> h_rgb;
  if (!LoadPpm(config.input_path, &width, &height, &h_rgb)) {
    std::cerr << "Failed to load input image: " << config.input_path << "\n";
    return EXIT_FAILURE;
  }

  size_t rgb_bytes = h_rgb.size();
  size_t gray_bytes = static_cast<size_t>(width) * height;
  std::vector<unsigned char> h_gray(gray_bytes, 0);

  unsigned char* d_rgb = nullptr;
  unsigned char* d_gray = nullptr;
  CHECK_CUDA(cudaMalloc(&d_rgb, rgb_bytes));
  CHECK_CUDA(cudaMalloc(&d_gray, gray_bytes));

  CHECK_CUDA(cudaMemcpy(d_rgb, h_rgb.data(), rgb_bytes,
                        cudaMemcpyHostToDevice));

  int blocks = (static_cast<int>(gray_bytes) + kThreadsPerBlock - 1) /
               kThreadsPerBlock;
  RgbToGrayscaleKernel<<<blocks, kThreadsPerBlock>>>(d_rgb, d_gray, width,
                                                     height);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_gray.data(), d_gray, gray_bytes,
                        cudaMemcpyDeviceToHost));

  if (!SavePgm(config.output_path, width, height, h_gray)) {
    std::cerr << "Failed to save output image: " << config.output_path << "\n";
    cudaFree(d_rgb);
    cudaFree(d_gray);
    return EXIT_FAILURE;
  }

  std::cout << "Converted " << config.input_path << " (" << width << "x"
            << height << ") -> " << config.output_path << "\n";

  cudaFree(d_rgb);
  cudaFree(d_gray);
  return EXIT_SUCCESS;
}
