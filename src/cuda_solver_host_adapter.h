#ifndef IFDDA_CUDA_SOLVER_HOST_ADAPTER_H
#define IFDDA_CUDA_SOLVER_HOST_ADAPTER_H

#include "cuda_data.h"

#include <limits>
#include <vector>

// The inverse driver accepts CPU std::complex buffers. Legacy and Chan
// adaptors translate their non-Green CPU arrays from cuComplex layout.
// The six Green CPU identity keys pass through unchanged to the shared driver,
// which resolves the separate resident GPU allocations before matrix products.
namespace cuda_solver_host_adapter {

inline bool checked_product(size_t a, size_t b, size_t *result) {
    if (!result || (b && a > std::numeric_limits<size_t>::max() / b))
        return false;
    *result = a * b;
    return true;
}

class Mirror {
public:
    bool initialize(const CUCOMPLEX *source, size_t count, CUCOMPLEX *output = nullptr) {
        if (!source || !count)
            return false;
        try {
            values_.resize(count);
        } catch (...) {
            return false;
        }
        for (size_t i = 0; i < count; ++i)
            values_[i] = COMPLEX(static_cast<REAL>(source[i].x),
                                 static_cast<REAL>(source[i].y));
        output_ = output;
        return true;
    }

    COMPLEX *data() { return values_.data(); }
    const COMPLEX *data() const { return values_.data(); }

    void publish() const {
        if (!output_)
            return;
        for (size_t i = 0; i < values_.size(); ++i) {
            output_[i].x = static_cast<CU_REAL>(values_[i].real());
            output_[i].y = static_cast<CU_REAL>(values_[i].imag());
        }
    }

private:
    std::vector<COMPLEX> values_;
    CUCOMPLEX *output_ = nullptr;
};

struct InverseDriverArguments {
    Mirror xr, work, field, incident, local_field, polarizability;

    // The original six COMPLEX* parameters are identity keys and are never
    // read, mirrored or converted: GPU-generated tensors need no CPU values.
    bool initialize_solver_buffers(CUCOMPLEX *host_xr, CUCOMPLEX *host_work,
                                   CUCOMPLEX *host_field,
                                   const CUCOMPLEX *host_incident,
                                   CUCOMPLEX *host_local_field,
                                   const CUCOMPLEX *host_polarizability,
                                   int nmax, int niso, int lda, int nlar,
                                   int ndim) {
        if (nmax <= 0 || niso <= 0 || lda <= 0 || nlar <= 0 || ndim <= 0)
            return false;
        size_t work_count = 0, polar_count = 0;
        if (!checked_product(size_t(lda), size_t(nlar), &work_count) ||
            !checked_product(size_t(nmax), size_t(niso), &polar_count) ||
            !checked_product(polar_count, size_t(niso), &polar_count))
            return false;
        return xr.initialize(host_xr, size_t(ndim), host_xr) &&
               work.initialize(host_work, work_count, host_work) &&
               field.initialize(host_field, size_t(ndim), host_field) &&
               incident.initialize(host_incident, size_t(ndim)) &&
               local_field.initialize(host_local_field, size_t(ndim), host_local_field) &&
               polarizability.initialize(host_polarizability, polar_count);
    }

    void publish() const {
        xr.publish();
        work.publish();
        field.publish();
        local_field.publish();
    }
};
} // namespace cuda_solver_host_adapter

#endif
