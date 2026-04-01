#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Core>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Eigen ↔ numpy helpers
// ---------------------------------------------------------------------------

// Convert a 1-D numpy array of shape (N,) to Eigen::Matrix<double, DIM, 1>.
// Throws if the array has the wrong size.
template <int DIM>
Eigen::Matrix<double, DIM, 1> np_to_vec(py::array_t<double> arr)
{
    auto info = arr.request();
    if (info.ndim != 1 || info.shape[0] != DIM)
        throw std::invalid_argument(
            "Expected 1-D array of length " + std::to_string(DIM) +
            ", got length " + std::to_string(info.ndim == 1 ? info.shape[0] : -1));
    Eigen::Matrix<double, DIM, 1> v;
    std::memcpy(v.data(), info.ptr, DIM * sizeof(double));
    return v;
}

// Convert a 1-D numpy array or None to Eigen::Matrix<double, DIM, 1>.
// If obj is None, returns the zero vector.
template <int DIM>
Eigen::Matrix<double, DIM, 1> np_to_vec_or_zero(py::object obj)
{
    if (obj.is_none())
        return Eigen::Matrix<double, DIM, 1>::Zero();
    return np_to_vec<DIM>(obj.cast<py::array_t<double>>());
}

// Return a numpy array that is a copy of an Eigen column vector.
template <int DIM>
py::array_t<double> vec_to_np(const Eigen::Matrix<double, DIM, 1>& v)
{
    py::array_t<double> arr(DIM);
    std::memcpy(arr.mutable_data(), v.data(), DIM * sizeof(double));
    return arr;
}

// Wrap an Eigen::VectorXd as a 1-D numpy array (copy).
inline py::array_t<double> vxd_to_np(const Eigen::VectorXd& v)
{
    py::array_t<double> arr(v.size());
    std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(double));
    return arr;
}

// Convert a 1-D numpy array to Eigen::VectorXd.
inline Eigen::VectorXd np_to_vxd(py::array_t<double> arr)
{
    auto info = arr.request();
    if (info.ndim != 1)
        throw std::invalid_argument("Expected 1-D array");
    Eigen::VectorXd v(info.shape[0]);
    std::memcpy(v.data(), info.ptr, info.shape[0] * sizeof(double));
    return v;
}

// ---------------------------------------------------------------------------
// PyTimeCost
// Wraps a Python callable with signature:
//   callable(times: np.ndarray) -> tuple[float, np.ndarray]
// where times has shape (N,) (decoded segment durations).
// Returns: (cost, gradient of shape (N,))
// ---------------------------------------------------------------------------
struct PyTimeCost
{
    py::object callable;

    double operator()(const std::vector<double>& times, Eigen::VectorXd& grad) const
    {
        py::gil_scoped_acquire gil;
        py::array_t<double> np_times(static_cast<py::ssize_t>(times.size()));
        std::memcpy(np_times.mutable_data(), times.data(), times.size() * sizeof(double));

        py::object result = callable(np_times);
        auto tup = result.cast<py::tuple>();
        if (tup.size() != 2)
            throw std::runtime_error("time_cost must return a 2-tuple (cost, grad)");

        double cost = tup[0].cast<double>();
        auto g = tup[1].cast<py::array_t<double>>();
        auto ginfo = g.request();
        if (ginfo.ndim != 1 || ginfo.shape[0] != static_cast<py::ssize_t>(times.size()))
            throw std::runtime_error("time_cost gradient has wrong shape");

        if (grad.size() != static_cast<Eigen::Index>(times.size()))
            grad.resize(times.size());
        std::memcpy(grad.data(), ginfo.ptr, times.size() * sizeof(double));
        return cost;
    }
};

// ---------------------------------------------------------------------------
// PyIntegralCost<DIM>
// Wraps a Python callable with signature:
//   callable(t, t_global, seg, step, p, v, a, j, s)
//       -> tuple[float, np.ndarray*6, float]
// where p,v,a,j,s each have shape (DIM,) and the returned tuple is:
//   (cost, gp, gv, ga, gj, gs, gt)
// ---------------------------------------------------------------------------
template <int DIM>
struct PyIntegralCost
{
    using Vec = Eigen::Matrix<double, DIM, 1>;
    py::object callable;

    double operator()(double t, double t_global, int seg, int step,
                      const Vec& p, const Vec& v, const Vec& a,
                      const Vec& j, const Vec& s,
                      Vec& gp, Vec& gv, Vec& ga, Vec& gj, Vec& gs,
                      double& gt) const
    {
        py::gil_scoped_acquire gil;

        auto to_np = [](const Vec& x) {
            py::array_t<double> arr(DIM);
            std::memcpy(arr.mutable_data(), x.data(), DIM * sizeof(double));
            return arr;
        };

        py::object result = callable(t, t_global, seg, step,
                                     to_np(p), to_np(v), to_np(a), to_np(j), to_np(s));
        auto tup = result.cast<py::tuple>();
        if (tup.size() != 7)
            throw std::runtime_error("integral_cost must return a 7-tuple (cost, gp, gv, ga, gj, gs, gt)");

        double cost = tup[0].cast<double>();

        auto from_np = [](Vec& out, py::handle h) {
            auto arr = h.cast<py::array_t<double>>();
            auto info = arr.request();
            if (info.shape[0] != DIM)
                throw std::runtime_error("integral_cost gradient has wrong dimension");
            std::memcpy(out.data(), info.ptr, DIM * sizeof(double));
        };

        from_np(gp, tup[1]);
        from_np(gv, tup[2]);
        from_np(ga, tup[3]);
        from_np(gj, tup[4]);
        from_np(gs, tup[5]);
        gt = tup[6].cast<double>();
        return cost;
    }
};
