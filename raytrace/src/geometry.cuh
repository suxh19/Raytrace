#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "helper_math.h" // cuda toolkit vector types

// CUDA device function qualifier for use in host/device compilable library code
#define CUDEV_FXN __host__ __device__


// Rotate vec around center using arbitrary axis
CUDEV_FXN float3 rotateAroundAxisRHS( const float3& vec, const float3& center, const float3& rotation_axis, const float& theta);
CUDEV_FXN float3 rotateAroundAxisAtOriginRHS(const float3& p, const float3& r, const float& t);

#endif // __GEOMETRY_H__
