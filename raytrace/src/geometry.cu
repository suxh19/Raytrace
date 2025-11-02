#include "geometry.cuh"

#include <cstdio>
#include "dev_intrinsics.cuh"


// Rotate vec around center using arbitrary axis
CUDEV_FXN float3 rotateAroundAxisRHS(const float3& p, const float3& q, const float3& r, const float& t) {
    // ASSUMES r IS NORMALIZED ALREADY
    // p - vector to rotate
    // q - center point
    // r - rotation axis
    // t - rotation angle
    // non-vectorized version
    //    x,y,z = p.(x,y,z)
    //    a,b,c = q.(x,y,z)
    //    u,v,w = r.(x,y,z)
    //
    //    /* (a*(fast_sq(v)+fast_sq(w)) - u*(b*v + c*w - u*x - v*y - w*z))*(1-fast_cosf(t)) + x*fast_cosf(t) + (-c*v + b*w - w*y + v*z)*fast_sinf(t), */
    //    /* (b*(fast_sq(u)+fast_sq(w)) - v*(a*u + c*w - u*x - v*y - w*z))*(1-fast_cosf(t)) + y*fast_cosf(t) + ( c*u - a*w + w*x - u*z)*fast_sinf(t), */
    //    /* (c*(fast_sq(u)+fast_sq(v)) - w*(a*u + b*v - u*x - v*y - w*z))*(1-fast_cosf(t)) + z*fast_cosf(t) + (-b*u + a*v - v*x + u*y)*fast_sinf(t) */

    float sptr, cptr;
    fast_sincosf(t, &sptr, &cptr);
    return make_float3(
            (q.x*(fast_sq(r.y)+fast_sq(r.z)) - r.x*(q.y*r.y + q.z*r.z - r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.x*cptr + (-q.z*r.y + q.y*r.z - r.z*p.y + r.y*p.z)*sptr,
            (q.y*(fast_sq(r.x)+fast_sq(r.z)) - r.y*(q.x*r.x + q.z*r.z - r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.y*cptr + ( q.z*r.x - q.x*r.z + r.z*p.x - r.x*p.z)*sptr,
            (q.z*(fast_sq(r.x)+fast_sq(r.y)) - r.z*(q.x*r.x + q.y*r.y - r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.z*cptr + (-q.y*r.x + q.x*r.y - r.y*p.x + r.x*p.y)*sptr
            );
}
CUDEV_FXN float3 rotateAroundAxisAtOriginRHS(const float3& p, const float3& r, const float& t) {
    // ASSUMES r IS NORMALIZED ALREADY and center is (0, 0, 0)
    // p - vector to rotate
    // r - rotation axis
    // t - rotation angle
    float sptr, cptr;
    fast_sincosf(t, &sptr, &cptr);
    return make_float3(
            (-r.x*(-r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.x*cptr + (-r.z*p.y + r.y*p.z)*sptr,
            (-r.y*(-r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.y*cptr + (+r.z*p.x - r.x*p.z)*sptr,
            (-r.z*(-r.x*p.x - r.y*p.y - r.z*p.z))*(1-cptr) + p.z*cptr + (-r.y*p.x + r.x*p.y)*sptr
            );
}
