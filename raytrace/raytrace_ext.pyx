ctypedef unsigned int uint

import numpy as np
cimport numpy as np

cdef extern from "vector_types.h":
    cdef struct float2:
        float x
        float y
    cdef struct float3:
        float x
        float y
        float z
    cdef struct uint2:
        uint x
        uint y
    cdef struct uint3:
        uint x
        uint y
        uint z

cdef extern from "src/raytrace.h":
    cdef void raytrace_c(
        float*        rpl,
        const float*  sources,
        const float*  dests,
        unsigned int  npts,
        float*        dens,
        float3        densStart,
        uint3         densSize,
        float3        densSpacing,
        float         stop_early
    )

def raytrace(sources, dests, vol, vol_start, vol_spacing, stop_early=-1):
    """Measures the pathlength along each ray specified by (source, dest) coordinate pairs through a voxelized
    volume.

    This function implements general raytracing between paired source/dest coordinates and can be used to
    implement arbitrary detector geometries completely from a high-level language with low-level execution

    Args:
        sources ([(x,y,z), ...]: list of 3d coordinates indicating ray startpoints
        dests   ([(x,y,z), ...]: list of 3d coordinates indicating ray endpoints
        vol (np.array): volume containing voxels through which to raytrace
        vol_start (float_x, float_y, float_z): coordinates of center of first voxel in vol
        vol_spacing (float_x, float_y, float_z): spacing between adjacent voxels in volume (assumes direct adjacency)
        stop_early: if <=0: trace through entire volume, if >0, stop after ray accumulates the value specified in this variable
            The output value in each detector element is not guaranteed to be equal to the value requested
    """
    for v in [sources, dests, vol]:
        assert isinstance(v, np.ndarray)
    assert sources.ndim == 2
    assert dests.ndim == 2
    assert vol.ndim == 3

    cdef uint  npts_ = dests.shape[0]
    cdef float[::1] rpl_ = np.zeros((npts_), dtype=np.float32)
    cdef float[:,::1] sources_ = np.ascontiguousarray(sources.astype(np.float32))
    cdef float[:,::1] dests_ = np.ascontiguousarray(dests.astype(np.float32))
    cdef float[:,:,::1] dens_ = np.ascontiguousarray(vol.astype(np.float32))
    cdef float3 dens_start_
    cdef uint3  dens_size_
    cdef float3 dens_spacing_

    # init structs/vects
    dens_start_.x, dens_start_.y, dens_start_.z = vol_start
    dens_size_.x, dens_size_.y, dens_size_.z = vol.shape[::-1]
    dens_spacing_.x, dens_spacing_.y, dens_spacing_.z = vol_spacing

    raytrace_c(
        rpl=&rpl_[0],
        sources=&sources_[0,0],
        dests=&dests_[0,0],
        npts=npts_,
        dens=&dens_[0,0,0],
        densStart=dens_start_,
        densSize=dens_size_,
        densSpacing=dens_spacing_,
        stop_early=stop_early
    )
    return np.asarray(rpl_)
