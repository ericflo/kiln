// CUDA -> HIP compatibility shim: `math_constants.h`.
//
// CUDA's `<math_constants.h>` defines `CUDART_*` float/double constants. HIP has
// no direct equivalent header, so define the subset kiln's kernels reference.
#pragma once

#include <math.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif
#ifndef CUDART_NAN_F
#define CUDART_NAN_F __int_as_float(0x7fffffff)
#endif
#ifndef CUDART_INF
#define CUDART_INF __longlong_as_double(0x7ff0000000000000ULL)
#endif
#ifndef CUDART_NAN
#define CUDART_NAN __longlong_as_double(0xfff8000000000000ULL)
#endif
#ifndef CUDART_ZERO_F
#define CUDART_ZERO_F 0.0f
#endif
