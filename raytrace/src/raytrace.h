#ifndef __RAYTRACE_HH__
#define __RAYTRACE_HH__

#include <iostream>
#include "vector_types.h"

void raytrace_c(
    float*        rpl,
    const float*  sources,
    const float*  dests,
    unsigned int  npts,
    float*        dens,
    float3        densStart,
    uint3         densSize,
    float3        densSpacing,
    float         stop_early=-1.f
    );

#endif //__RAYTRACE_HH__

