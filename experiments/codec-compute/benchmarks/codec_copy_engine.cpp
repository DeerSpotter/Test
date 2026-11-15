#include <cuda.h>
#include "nvcuvid.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>
#include <string>

using Clock = std::chrono::steady_clock;
constexpr int width = 256, height = 256, luma = width * height;
constexpr int frame_bytes = luma * 3 / 2;
constexpr int decode_surfaces = 12;

struct Stats {
    int displayed, copied, decode_submissions, submissions_with_copy_pending;
    double map_ms, sync_wait_ms;
};

static void check(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* name = nullptr;
    cuGetErrorName(result, &name);
    throw std::runtime_error(std::string(operation) + ": " +
                             (name ? name : std::to_string(result)));
}

static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Slot {
    uint8_t* host = nullptr;
    CUstream stream = nullptr;
    CUevent done = nullptr;
    CUdeviceptr mapped = 0;
};

struct Runner {
    explicit Runner(int capacity) : capacity(capacity) {}

    int capacity;
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUvideodecoder decoder = nullptr;
    CUvideoparser parser = nullptr;
    Slot slots[2];
    Stats stats{};
    std::string callback_error;
    bool asynchronous = false;
    bool sums = false;

    void init() {
        check(cuInit(0), "cuInit");
        check(cuDeviceGet(&device, 0), "cuDeviceGet");
        check(cuDevicePrimaryCtxRetain(&context, device), "retain CUDA context");
        check(cuCtxSetCurrent(context), "set CUDA context");
        for (auto& slot : slots) {
            check(cuMemAllocHost(reinterpret_cast<void**>(&slot.host),
                                 size_t(capacity) * frame_bytes), "allocate pinned host buffer");
            check(cuStreamCreate(&slot.stream, CU_STREAM_NON_BLOCKING), "create copy stream");
            check(cuEventCreate(&slot.done, CU_EVENT_DISABLE_TIMING | CU_EVENT_BLOCKING_SYNC),
                  "create copy event");
        }
    }

    ~Runner() {
        clear_session();
        for (auto& slot : slots) {
            if (slot.done) cuEventDestroy(slot.done);
            if (slot.stream) cuStreamDestroy(slot.stream);
            if (slot.host) cuMemFreeHost(slot.host);
        }
        if (context) cuDevicePrimaryCtxRelease(device);
    }

    void clear_session() {
        for (auto& slot : slots) {
            if (slot.mapped) {
                cuStreamSynchronize(slot.stream);
                cuvidUnmapVideoFrame(decoder, slot.mapped);
                slot.mapped = 0;
            }
        }
        if (parser) {
            cuvidDestroyVideoParser(parser);
            parser = nullptr;
        }
        if (decoder) {
            cuvidDestroyDecoder(decoder);
            decoder = nullptr;
        }
    }

    void retire(Slot& slot) {
        if (!slot.mapped) return;
        auto start = Clock::now();
        check(cuEventSynchronize(slot.done), "wait for completed host copy");
        stats.sync_wait_ms += milliseconds(start);
        check(cuvidUnmapVideoFrame(decoder, slot.mapped), "release mapped frame");
        slot.mapped = 0;
    }

    static int CUDAAPI sequence(void* opaque, CUVIDEOFORMAT* format) {
        auto& self = *static_cast<Runner*>(opaque);
        try {
            if (format->coded_width != width || format->coded_height != height ||
                format->bit_depth_luma_minus8 != 0 || format->chroma_format != cudaVideoChromaFormat_420)
                throw std::runtime_error("Expected 256x256 8-bit 4:2:0 input");
            if (self.decoder) return decode_surfaces;
            CUVIDDECODECREATEINFO info{};
            info.CodecType = format->codec;
            info.ChromaFormat = format->chroma_format;
            info.ulWidth = info.ulTargetWidth = width;
            info.ulHeight = info.ulTargetHeight = height;
            info.ulNumDecodeSurfaces = decode_surfaces;
            info.ulNumOutputSurfaces = 2;
            info.ulCreationFlags = cudaVideoCreate_PreferCUVID;
            info.OutputFormat = cudaVideoSurfaceFormat_NV12;
            info.DeinterlaceMode = cudaVideoDeinterlaceMode_Weave;
            check(cuvidCreateDecoder(&self.decoder, &info), "create NVDEC decoder");
            return info.ulNumDecodeSurfaces;
        } catch (const std::exception& error) {
            self.callback_error = error.what();
            return 0;
        }
    }

    static int CUDAAPI decode_picture(void* opaque, CUVIDPICPARAMS* picture) {
        auto& self = *static_cast<Runner*>(opaque);
        try {
            bool copying = false;
            for (auto& slot : self.slots) {
                if (!slot.mapped) continue;
                CUresult state = cuEventQuery(slot.done);
                if (state == CUDA_ERROR_NOT_READY) copying = true;
                else check(state, "query copy completion");
            }
            self.stats.submissions_with_copy_pending += copying;
            ++self.stats.decode_submissions;
            check(cuvidDecodePicture(self.decoder, picture), "submit NVDEC picture");
            return 1;
        } catch (const std::exception& error) {
            self.callback_error = error.what();
            return 0;
        }
    }

    static int CUDAAPI display(void* opaque, CUVIDPARSERDISPINFO* frame) {
        auto& self = *static_cast<Runner*>(opaque);
        try {
            if (!frame) return 1;
            int displayed = self.stats.displayed++;
            if (self.sums && displayed % 3 != 1) return 1;
            int index = self.stats.copied;
            if (index / 2 >= self.capacity)
                throw std::runtime_error("Pinned output buffer is full");
            Slot& slot = self.slots[index % 2];
            self.retire(slot);
            CUVIDPROCPARAMS params{};
            params.progressive_frame = frame->progressive_frame;
            params.top_field_first = frame->top_field_first;
            params.unpaired_field = 1;
            unsigned pitch = 0;
            auto start = Clock::now();
            check(cuvidMapVideoFrame(self.decoder, frame->picture_index, &slot.mapped,
                                     &pitch, &params), "map NVDEC frame");
            self.stats.map_ms += milliseconds(start);
            CUDA_MEMCPY2D copy{};
            copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            copy.srcDevice = slot.mapped;
            copy.srcPitch = pitch;
            copy.dstMemoryType = CU_MEMORYTYPE_HOST;
            copy.dstHost = slot.host + size_t(index / 2) * frame_bytes;
            copy.dstPitch = copy.WidthInBytes = width;
            copy.Height = height + height / 2;
            check(cuMemcpy2DAsync(&copy, slot.stream), "copy NV12 frame into pinned memory");
            check(cuEventRecord(slot.done, slot.stream), "record copy completion");
            ++self.stats.copied;
            if (!self.asynchronous) self.retire(slot);
            return 1;
        } catch (const std::exception& error) {
            self.callback_error = error.what();
            return 0;
        }
    }

    void run(const uint8_t* data, size_t bytes, bool pipeline, bool sums_only) {
        check(cuCtxSetCurrent(context), "set CUDA context");
        stats = {};
        callback_error.clear();
        asynchronous = pipeline;
        sums = sums_only;
        CUVIDPARSERPARAMS params{};
        params.CodecType = cudaVideoCodec_H264;
        params.ulMaxNumDecodeSurfaces = decode_surfaces;
        params.ulMaxDisplayDelay = 2;
        params.pUserData = this;
        params.pfnSequenceCallback = sequence;
        params.pfnDecodePicture = decode_picture;
        params.pfnDisplayPicture = display;
        check(cuvidCreateVideoParser(&parser, &params), "create H.264 parser");
        CUVIDSOURCEDATAPACKET packet{};
        packet.payload = data;
        packet.payload_size = bytes;
        CUresult status = cuvidParseVideoData(parser, &packet);
        if (!callback_error.empty()) throw std::runtime_error(callback_error);
        check(status, "parse H.264");
        packet = {};
        packet.flags = CUVID_PKT_ENDOFSTREAM;
        status = cuvidParseVideoData(parser, &packet);
        if (!callback_error.empty()) throw std::runtime_error(callback_error);
        check(status, "flush H.264");
        for (auto& slot : slots) retire(slot);
        clear_session();
    }
};

extern "C" Runner* make_runner(int capacity, char* error, size_t size) {
    Runner* runner = nullptr;
    try {
        runner = new Runner(capacity);
        runner->init();
        return runner;
    } catch (const std::exception& failure) {
        std::snprintf(error, size, "%s", failure.what());
        delete runner;
        return nullptr;
    }
}

extern "C" int run_decoder(Runner* runner, const uint8_t* data, size_t bytes,
                            int async_copy, int sums, Stats* stats, char* error, size_t size) {
    try {
        runner->run(data, bytes, async_copy, sums);
        *stats = runner->stats;
        return 0;
    } catch (const std::exception& failure) {
        runner->clear_session();
        std::snprintf(error, size, "%s", failure.what());
        return 1;
    }
}

extern "C" int async_engines(Runner* runner) {
    int count = 0;
    if (cuDeviceGetAttribute(&count, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, runner->device) != CUDA_SUCCESS)
        return -1;
    return count;
}

extern "C" const uint8_t* output_slot(Runner* runner, int index) {
    return runner->slots[index].host;
}

extern "C" void destroy_runner(Runner* runner) {
    delete runner;
}

extern "C" void add_on_cpu(Runner* runner, int count, uint8_t* output, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
    for (int f = 0; f < count; ++f) {
        const auto* a = runner->slots[0].host + size_t(f) * frame_bytes;
        const auto* b = runner->slots[1].host + size_t(f) * frame_bytes;
        auto* dst = output + size_t(f) * frame_bytes;
        for (int i=0; i<luma; i+=32) {
            __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+i));
            __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b+i));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst+i), _mm256_adds_epu8(x,y));
        }
        std::memcpy(dst+luma, a+luma, frame_bytes-luma);
    }
}
