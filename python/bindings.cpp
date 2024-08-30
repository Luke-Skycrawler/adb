#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "collision.h"
#include <ipc/distance/point_triangle.hpp>
#include <ipc/distance/edge_edge.hpp>
#include <ipc/utils/eigen_ext.hpp>
#include <ipc/distance/edge_edge_mollifier.hpp>
#include <ipc/friction/smooth_friction_mollifier.hpp>
#include <tuple>
namespace py = pybind11;
using namespace std;

tuple<mat12, vec12> ipc_hess_pt_12x12(
    array<vec3, 4> pt, array<int, 4> ij, ipc::PointTriangleDistanceType pt_type, double dist);
tuple<mat12, vec12, double> ipc_hess_ee_12x12(
    array<vec3, 4> ee, array<int, 4> ij,
    ipc::EdgeEdgeDistanceType ee_type, double dist);


PYBIND11_MODULE(abdtk, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: shaysweep

        .. autosummary::
           :toctree: _generate


    )pbdoc";
    py::enum_<ipc::PointTriangleDistanceType>(m, "PointTriangleDistanceType")
        .value("P_T0", ipc::PointTriangleDistanceType::P_T0)
        .value("P_T1", ipc::PointTriangleDistanceType::P_T1)
        .value("P_T2", ipc::PointTriangleDistanceType::P_T2)
        .value("P_E0", ipc::PointTriangleDistanceType::P_E0)
        .value("P_E1", ipc::PointTriangleDistanceType::P_E1)
        .value("P_E2", ipc::PointTriangleDistanceType::P_E2)
        .value("P_T", ipc::PointTriangleDistanceType::P_T)
        .value("AUTO", ipc::PointTriangleDistanceType::AUTO)
        .export_values();

    py::enum_<ipc::EdgeEdgeDistanceType>(m, "EdgeEdgeDistanceType")
        .value("EA0_EB0", ipc::EdgeEdgeDistanceType::EA0_EB0)
        .value("EA0_EB1", ipc::EdgeEdgeDistanceType::EA0_EB1)
        .value("EA1_EB0", ipc::EdgeEdgeDistanceType::EA1_EB0)
        .value("EA1_EB1", ipc::EdgeEdgeDistanceType::EA1_EB1)
        .value("EA_EB0", ipc::EdgeEdgeDistanceType::EA_EB0)
        .value("EA_EB1", ipc::EdgeEdgeDistanceType::EA_EB1)
        .value("EA0_EB", ipc::EdgeEdgeDistanceType::EA0_EB)
        .value("EA1_EB", ipc::EdgeEdgeDistanceType::EA1_EB)
        .value("EA_EB", ipc::EdgeEdgeDistanceType::EA_EB)
        .value("AUTO", ipc::EdgeEdgeDistanceType::AUTO)
        .export_values();

    m.def("ipc_hess_pt_12x12", &ipc_hess_pt_12x12, "Computes the hessian and gradient of the point-triangle distance", py::arg("pt"), py::arg("ij"), py::arg("pt_type"), py::arg("dist"));
    m.def("ipc_hess_ee_12x12", &ipc_hess_ee_12x12, "Computes the hessian and gradient of the edge-edge distance", py::arg("ee"), py::arg("ij"), py::arg("ee_type"), py::arg("dist"));
}