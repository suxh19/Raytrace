#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

extern "C" void raytrace_cuda_forward(
    float* rpl,
    const float* sources,
    const float* dests,
    unsigned int npts,
    const float* vol,
    unsigned int D, unsigned int H, unsigned int W,
    float sx, float sy, float sz,
    float spx, float spy, float spz,
    float stop_early,
    cudaStream_t stream
);

static void check_inputs(const torch::Tensor& t, bool expect_cuda, const char* name, std::initializer_list<int64_t> last_dim = {}) {
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (expect_cuda) {
        TORCH_CHECK(t.is_cuda(), name, " must be CUDA tensor");
    } else {
        TORCH_CHECK(!t.is_cuda(), name, " must be CPU tensor");
    }
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    if (last_dim.size() > 0) {
        auto it = last_dim.begin();
        int64_t k = *it;
        if (k > 0) {
            TORCH_CHECK(t.size(-1) == k, name, " last dim must be ", k);
        }
    }
}

torch::Tensor raytrace_forward(
    torch::Tensor sources,
    torch::Tensor dests,
    torch::Tensor vol,
    float sx, float sy, float sz,
    float spx, float spy, float spz,
    float stop_early
) {
    check_inputs(sources, true, "sources", {3});
    check_inputs(dests,   true, "dests",   {3});
    check_inputs(vol,     true, "vol");
    TORCH_CHECK(sources.dim() == 2 && sources.size(1) == 3, "sources must have shape [N,3]");
    TORCH_CHECK(dests.dim() == 2 && dests.size(1) == 3,     "dests must have shape [N,3]");
    TORCH_CHECK(vol.dim() == 3, "vol must have shape [D,H,W]");

    const auto N = static_cast<unsigned int>(sources.size(0));
    const auto D = static_cast<unsigned int>(vol.size(0));
    const auto H = static_cast<unsigned int>(vol.size(1));
    const auto W = static_cast<unsigned int>(vol.size(2));

    auto opts = sources.options();
    auto out = torch::empty({static_cast<long long>(N)}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    raytrace_cuda_forward(
        out.data_ptr<float>(),
        sources.data_ptr<float>(),
        dests.data_ptr<float>(),
        N,
        vol.data_ptr<float>(),
        D, H, W,
        sx, sy, sz,
        spx, spy, spz,
        stop_early,
        stream
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("raytrace_forward", &raytrace_forward, "Raytrace forward (CUDA)");
}
