#include <MeshFEM/Geometry.hh>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>
namespace py = pybind11;

PYBIND11_MODULE(vector_operations, m) {
    m.def("getPerpendicularVector",      &getPerpendicularVector<double>,      py::arg("v"));
    m.def("curvatureBinormal",           &curvatureBinormal<double>,           py::arg("e0"),           py::arg("e1"));
    m.def("rotatedVector",               &rotatedVector<double>,               py::arg("sinThetaAxis"), py::arg("cosTheta"), py::arg("v"));
    m.def("parallelTransportNormalized", &parallelTransportNormalized<double>, py::arg("t0"),           py::arg("t1"),       py::arg("v"));
    m.def("parallelTransport",           &parallelTransport<double>,           py::arg("t0"),           py::arg("t1"),       py::arg("v"));
    m.def("angle",                       [](const Vector3D &v1, const Vector3D &v2) { return angle(v1, v2); }, py::arg("v1"), py::arg("v2"));
    m.def("signedAngle",                 [](const Vector3D &v1, const Vector3D &v2, const Vector3D &n) { return signedAngle(v1, v2, n); }, py::arg("v1"), py::arg("v2"), py::arg("n"));
}
