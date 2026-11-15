#include <cuda.h>
#include "nvOpticalFlowCuda.h"
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int size = 256;
constexpr int grid = 4;
constexpr int flow_size = size / grid;

void cuda_check(CUresult code, const char* operation) {
    if (code == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(code, &name);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name ? name : std::to_string(code)));
}

void of_check(NV_OF_STATUS code, const char* operation) {
    if (code != NV_OF_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": NVOF status " +
                                 std::to_string(code));
}

struct Session {
    void* driver = nullptr;
    CUdevice device = 0;
    CUcontext context = nullptr;
    NvOFHandle handle = nullptr;
    std::array<NvOFGPUBufferHandle, 3> buffers{};
    NV_OF_CUDA_API_FUNCTION_LIST api{};

    ~Session() {
        for (auto buffer : buffers)
            if (buffer) api.nvOFDestroyGPUBufferCuda(buffer);
        if (handle) api.nvOFDestroy(handle);
        if (context) cuDevicePrimaryCtxRelease(device);
        if (driver) dlclose(driver);
    }

    void init() {
        cuda_check(cuInit(0), "initialize CUDA driver");
        cuda_check(cuDeviceGet(&device, 0), "select device");
        cuda_check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
        cuda_check(cuCtxSetCurrent(context), "set context");
        driver = dlopen("libnvidia-opticalflow.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!driver) throw std::runtime_error(dlerror());
        auto entry = reinterpret_cast<decltype(&NvOFAPICreateInstanceCuda)>(
            dlsym(driver, "NvOFAPICreateInstanceCuda"));
        if (!entry) throw std::runtime_error(dlerror());
        of_check(entry(NV_OF_API_VERSION, &api), "load optical flow interface");
        of_check(api.nvCreateOpticalFlowCuda(context, &handle), "create optical flow session");
        NV_OF_INIT_PARAMS params{};
        params.width = params.height = size;
        params.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_4;
        params.mode = NV_OF_MODE_OPTICALFLOW;
        params.perfLevel = NV_OF_PERF_LEVEL_SLOW;
        of_check(api.nvOFInit(handle, &params), "initialize 4x4 optical flow");
        const NV_OF_BUFFER_DESCRIPTOR frame{size, size, NV_OF_BUFFER_USAGE_INPUT,
                                           NV_OF_BUFFER_FORMAT_GRAYSCALE8};
        const NV_OF_BUFFER_DESCRIPTOR result{flow_size, flow_size, NV_OF_BUFFER_USAGE_OUTPUT,
                                            NV_OF_BUFFER_FORMAT_SHORT2};
        for (int index = 0; index < 3; ++index) {
            of_check(api.nvOFCreateGPUBufferCuda(handle, index == 2 ? &result : &frame,
                     NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &buffers[index]), "allocate OF buffer");
        }
    }

    void transfer(int index, void* host, size_t row_bytes, size_t rows, bool upload) {
        NV_OF_CUDA_BUFFER_STRIDE_INFO stride{};
        of_check(api.nvOFGPUBufferGetStrideInfo(buffers[index], &stride), "read buffer pitch");
        auto data = api.nvOFGPUBufferGetCUdeviceptr(buffers[index]);
        if (!data) throw std::runtime_error("No GPU pointer for optical flow buffer");

        CUDA_MEMCPY2D copy{};
        copy.WidthInBytes = row_bytes;
        copy.Height = rows;
        if (upload) {
            copy.srcMemoryType = CU_MEMORYTYPE_HOST;
            copy.srcHost = host;
            copy.srcPitch = row_bytes;
            copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.dstDevice = data;
            copy.dstPitch = stride.strideInfo[0].strideXInBytes;
        } else {
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = data;
            copy.srcPitch = stride.strideInfo[0].strideXInBytes;
            copy.dstMemoryType = CU_MEMORYTYPE_HOST;
            copy.dstHost = host;
            copy.dstPitch = row_bytes;
        }
        cuda_check(cuMemcpy2D(&copy), "transfer optical flow buffer");
    }

    std::vector<NV_OF_FLOW_VECTOR> calculate(std::vector<uint8_t>& a, std::vector<uint8_t>& b) {
        transfer(0, a.data(), size, size, true);
        transfer(1, b.data(), size, size, true);
        NV_OF_EXECUTE_INPUT_PARAMS inputs{};
        inputs.inputFrame = buffers[0];
        inputs.referenceFrame = buffers[1];
        inputs.disableTemporalHints = NV_OF_TRUE;
        NV_OF_EXECUTE_OUTPUT_PARAMS output{};
        output.outputBuffer = buffers[2];
        of_check(api.nvOFExecute(handle, &inputs, &output), "execute optical flow");
        cuda_check(cuCtxSynchronize(), "complete optical flow operation");
        std::vector<NV_OF_FLOW_VECTOR> vectors(flow_size * flow_size);
        transfer(2, vectors.data(), flow_size * sizeof(NV_OF_FLOW_VECTOR), flow_size, false);
        return vectors;
    }
};

std::vector<uint8_t> texture() {
    std::vector<uint8_t> noise(size * size), output(size * size);
    uint32_t state = 0x8126a53b;
    for (auto& pixel : noise) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        pixel = static_cast<uint8_t>(state);
    }
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            auto at = [&](int a, int b) { return noise[((b + size) % size) * size + (a + size) % size]; };
            output[y*size + x] = static_cast<uint8_t>(
                (4 * at(x,y) + at(x-1,y) + at(x+1,y) + at(x,y-1) + at(x,y+1)) / 8);
        }
    return output;
}

std::vector<uint8_t> shift(const std::vector<uint8_t>& image, int dx, int dy) {
    std::vector<uint8_t> result(image.size());
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            result[y*size + x] = image[((y - dy + size) % size) * size + (x - dx + size) % size];
    return result;
}

int main() {
    try {
        Session session;
        session.init();
        auto source = texture();
        bool passed = true;
        std::cout << "{\"image_size\":" << size << ",\"grid\":" << grid
                  << ",\"cases\":[";
        const std::array<std::array<int,2>,4> shifts{{{0,0},{8,-4},{-12,8},{4,12}}};
        for (size_t index = 0; index < shifts.size(); ++index) {
            const auto [dx, dy] = shifts[index];
            auto destination = shift(source, dx, dy);
            auto vectors = session.calculate(source, destination);
            std::vector<int16_t> xs, ys;
            int exact = 0;
            int close = 0;
            constexpr int margin = 10;
            for (int y = margin; y < flow_size - margin; ++y)
                for (int x = margin; x < flow_size - margin; ++x) {
                    auto vector = vectors[y * flow_size + x];
                    xs.push_back(vector.flowx);
                    ys.push_back(vector.flowy);
                    int error_x = std::abs(vector.flowx - dx * 32);
                    int error_y = std::abs(vector.flowy - dy * 32);
                    exact += error_x == 0 && error_y == 0;
                    close += error_x <= 32 && error_y <= 32;
                }
            std::sort(xs.begin(), xs.end());
            std::sort(ys.begin(), ys.end());
            double x = xs[xs.size()/2] / 32.0;
            double y = ys[ys.size()/2] / 32.0;
            bool good = std::abs(x - dx) <= 0.5 && std::abs(y - dy) <= 0.5
                        && close * 10 >= 9 * static_cast<int>(xs.size());
            passed = passed && good;
            if (index) std::cout << ',';
            std::cout << "{\"expected\":[" << dx << ',' << dy << "],\"median\":["
                      << x << ',' << y << "],\"blocks\":" << xs.size()
                      << ",\"exact\":" << exact << ",\"within_one_pixel\":" << close
                      << ",\"passed\":" << (good ? "true" : "false") << '}';
        }
        std::cout << "],\"passed\":" << (passed ? "true" : "false") << "}\n";
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
