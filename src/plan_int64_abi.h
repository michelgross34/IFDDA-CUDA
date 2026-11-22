#ifndef IFDDA_PLAN_INT64_ABI_H
#define IFDDA_PLAN_INT64_ABI_H

#include <cstdint>

/* Linux x86_64 uses long for int64_t; Windows MinGW uses long long. */
#if defined(IFDDA_WSL_ABI_INT64_PLAN)
using IFDDA_PLAN_INT64 = std::int64_t;
#else
using IFDDA_PLAN_INT64 = long long;
#endif

#endif
