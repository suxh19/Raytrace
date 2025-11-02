 #include <cuda_runtime.h>
 #include <cmath>
 #include <cassert>
 #include <cstdint>

 static __device__ __forceinline__ float length3(const float3& v) {
     return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
 }

 static __device__ float siddonRPL(
     float3 source,
     float3 dest,
     float3 start,
     uint3 size,
     float3 spacing,
     cudaTextureObject_t texDens,
     float stop_early
 ) {
     dest.x += 1e-12f; dest.y += 1e-12f; dest.z += 1e-12f;
     float3 diff; diff.x = dest.x - source.x; diff.y = dest.y - source.y; diff.z = dest.z - source.z;

     float a_min, a_max;
     float3 alpha_min, alpha_max;
     {
         float3 end;
         end.x = start.x + spacing.x*((float)size.x) - 1.f;
         end.y = start.y + spacing.y*((float)size.y) - 1.f;
         end.z = start.z + spacing.z*((float)size.z) - 1.f;

         float3 a_first, a_last;
         a_first.x = (start.x - source.x - 0.5f*spacing.x) / diff.x;
         a_first.y = (start.y - source.y - 0.5f*spacing.y) / diff.y;
         a_first.z = (start.z - source.z - 0.5f*spacing.z) / diff.z;
         a_last.x  = (end.x   - source.x + 0.5f*spacing.x) / diff.x;
         a_last.y  = (end.y   - source.y + 0.5f*spacing.y) / diff.y;
         a_last.z  = (end.z   - source.z + 0.5f*spacing.z) / diff.z;

         alpha_min.x = fminf(a_first.x, a_last.x);
         alpha_min.y = fminf(a_first.y, a_last.y);
         alpha_min.z = fminf(a_first.z, a_last.z);
         alpha_max.x = fmaxf(a_first.x, a_last.x);
         alpha_max.y = fmaxf(a_first.y, a_last.y);
         alpha_max.z = fmaxf(a_first.z, a_last.z);

         a_min = fmaxf(0.f, fmaxf(fmaxf(alpha_min.x, alpha_min.y), alpha_min.z));
         a_max = fminf(1.f, fminf(fminf(alpha_max.x, alpha_max.y), alpha_max.z));
     }

     float rpl = 0.f;
     if (!(a_min >= a_max)) {
         float d12 = length3(diff);
         float step_x = fabsf(spacing.x/diff.x);
         float step_y = fabsf(spacing.y/diff.y);
         float step_z = fabsf(spacing.z/diff.z);

         float alpha = a_min;
         float alpha_x = a_min;
         float alpha_y = a_min;
         float alpha_z = a_min;
         float nextalpha;
         float alpha_mid;
         const int max_iters = 5000;
         const float intersect_min = 0.001f*fminf(fminf(spacing.x, spacing.y), spacing.z);
         int iter = 0;
         while (alpha < a_max && ++iter < max_iters) {
             bool valid_x = (alpha_x >= 0.0f && alpha_x < 1.0f);
             bool valid_y = (alpha_y >= 0.0f && alpha_y < 1.0f);
             bool valid_z = (alpha_z >= 0.0f && alpha_z < 1.0f);
             if (!(valid_x || valid_y || valid_z)) { break; }
             if (valid_x && (!valid_y || alpha_x <= alpha_y) && (!valid_z || alpha_x <= alpha_z)) {
                 nextalpha = alpha_x; alpha_x += step_x;
             } else if (valid_y && (!valid_x || alpha_y <= alpha_x) && (!valid_z || alpha_y <= alpha_z)) {
                 nextalpha = alpha_y; alpha_y += step_y;
             } else if (valid_z && (!valid_x || alpha_z <= alpha_x) && (!valid_y || alpha_z <= alpha_y)) {
                 nextalpha = alpha_z; alpha_z += step_z;
             }
             float intersection = fabsf(d12*(nextalpha-alpha));
             if (intersection >= intersect_min) {
                 alpha_mid = (nextalpha + alpha)*0.5f;
                 float fetchX = (source.x + alpha_mid*diff.x - start.x) / spacing.x;
                 float fetchY = (source.y + alpha_mid*diff.y - start.y) / spacing.y;
                 float fetchZ = (source.z + alpha_mid*diff.z - start.z) / spacing.z;
                 rpl += intersection * tex3D<float>(texDens, fetchX, fetchY, fetchZ);
                 if (stop_early >= 0.f && rpl > stop_early) { break; }
             }
             alpha = nextalpha;
         }
     }
     if (!isfinite(rpl)) { rpl = 0.f; }
     return rpl;
 }

 __global__ void cudaRayTrace_kernel(
     float*  rpl,
     const float*  sources,
     const float*  dests,
     unsigned int  npts,
     float3  densStart,
     uint3   densSize,
     float3  densSpacing,
     cudaTextureObject_t texDens,
     float   stop_early
 ) {
     unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
     if (tid >= npts) return;

     float3 source = {sources[3*tid + 0], sources[3*tid + 1], sources[3*tid + 2]};
     float3 dest   = {dests  [3*tid + 0], dests  [3*tid + 1], dests  [3*tid + 2]};

     rpl[tid] = siddonRPL(source, dest, densStart, densSize, densSpacing, texDens, stop_early);
 }

 struct TextureCleanupCtx {
     cudaArray_t arr;
     cudaTextureObject_t tex;
 };

 static void CUDART_CB cleanup_callback(void* userData) {
     TextureCleanupCtx* ctx = reinterpret_cast<TextureCleanupCtx*>(userData);
     if (ctx) {
         if (ctx->tex) cudaDestroyTextureObject(ctx->tex);
         if (ctx->arr) cudaFreeArray(ctx->arr);
         delete ctx;
     }
 }

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
 ) {
     cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
     cudaArray_t d_densArr;
     cudaExtent extent = make_cudaExtent(W, H, D);
     cudaMalloc3DArray(&d_densArr, &desc, extent);

     cudaMemcpy3DParms copyParams = {0};
     copyParams.srcPtr = make_cudaPitchedPtr((void*)vol, W*sizeof(float), W, H);
     copyParams.dstArray = d_densArr;
     copyParams.kind = cudaMemcpyDeviceToDevice;
     copyParams.extent = extent;
     cudaMemcpy3DAsync(&copyParams, stream);

     cudaResourceDesc resDesc; memset(&resDesc, 0, sizeof(resDesc));
     resDesc.resType = cudaResourceTypeArray;
     resDesc.res.array.array = d_densArr;
     cudaTextureDesc texDesc; memset(&texDesc, 0, sizeof(texDesc));
     texDesc.normalizedCoords = false;
     texDesc.filterMode = cudaFilterModeLinear;
     texDesc.addressMode[0] = cudaAddressModeBorder;
     texDesc.addressMode[1] = cudaAddressModeBorder;
     texDesc.addressMode[2] = cudaAddressModeBorder;
     texDesc.readMode = cudaReadModeElementType;
     cudaTextureObject_t texDens;
     cudaCreateTextureObject(&texDens, &resDesc, &texDesc, nullptr);

     uint3 densSize = make_uint3(W, H, D);
     float3 densStart = make_float3(sx, sy, sz);
     float3 densSpacing = make_float3(spx, spy, spz);

     int block = 256;
     int grid = (int)((npts + block - 1) / block);
     cudaRayTrace_kernel<<<grid, block, 0, stream>>>(
         rpl, sources, dests, npts, densStart, densSize, densSpacing, texDens, stop_early
     );

     TextureCleanupCtx* ctx = new TextureCleanupCtx{d_densArr, texDens};
     cudaLaunchHostFunc(stream, cleanup_callback, ctx);
 }
