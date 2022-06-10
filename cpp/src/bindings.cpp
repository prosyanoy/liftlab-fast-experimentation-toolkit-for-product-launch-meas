#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>

#include "liftlab/resample.hpp"

namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

template <class Kernel>
py::array_t<double> run(Array control, Array treatment, std::size_t n, std::uint64_t seed,
                        unsigned threads, Kernel kernel) {
    if (control.ndim() != 1 || treatment.ndim() != 1)
        throw std::invalid_argument("inputs must be 1-D arrays");
    std::vector<double> result;
    {
        py::gil_scoped_release release;  // heavy loop runs without the GIL
        result = kernel(control.data(), static_cast<std::size_t>(control.size()), treatment.data(),
                        static_cast<std::size_t>(treatment.size()), n, seed, threads);
    }
    py::array_t<double> out(static_cast<py::ssize_t>(result.size()));
    std::copy(result.begin(), result.end(), out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_cpp, m) {
    m.doc() = "Multithreaded bootstrap / permutation kernels for LiftLab";
    m.def(
        "bootstrap_mean_diff",
        [](Array c, Array t, std::size_t n_boot, std::uint64_t seed, unsigned threads) {
            return run(c, t, n_boot, seed, threads, liftlab::bootstrap_mean_diff);
        },
        py::arg("control"), py::arg("treatment"), py::arg("n_boot"), py::arg("seed") = 0,
        py::arg("threads") = 0,
        "Bootstrap distribution of mean(treatment) - mean(control). threads=0 uses all cores.");
    m.def(
        "permutation_mean_diff",
        [](Array c, Array t, std::size_t n_perm, std::uint64_t seed, unsigned threads) {
            return run(c, t, n_perm, seed, threads, liftlab::permutation_mean_diff);
        },
        py::arg("control"), py::arg("treatment"), py::arg("n_perm"), py::arg("seed") = 0,
        py::arg("threads") = 0,
        "Permutation null distribution of mean(treatment) - mean(control).");
}
