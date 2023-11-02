#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

#include <OffscreenRenderer/Shader.hh>
#include <OffscreenRenderer/OpenGLContext.hh>
#include <OffscreenRenderer/Buffers.hh>

namespace py = pybind11;

// Apply F::run<T>(args) for each T in Ts
template<template<typename> class F, typename... Ts>
struct MetaMap;

template<template<typename> class F>
struct MetaMap<F> {
    template<typename... Args>
    static void run(Args &&... /* args */) { }
};

template<template<typename> class F, typename T, typename... Ts>
struct MetaMap<F, T, Ts...> {
    template<typename... Args>
    static void run(Args &&... args) {
        F<T>::run(std::forward<Args>(args)...);
        MetaMap<F, Ts...>::run(std::forward<Args>(args)...);
    }
};

template<typename T>
struct BindSetUniform {
    template<class PyShader>
    static void run(PyShader &pyShader) {
        pyShader.def("setUniform", &Shader::setUniform<T>, py::arg("name"), py::arg("val"));
    }
};

template<typename T>
struct BindSetConstAttribute {
    template<class PyVAO>
    static void run(PyVAO &pyVAO) {
        pyVAO.def("setConstantAttribute", &VertexArrayObject::setConstantAttribute<T>, py::arg("attribIdx"), py::arg("val"));
    }
};

#include "py_gl_enum.inl"

PYBIND11_MODULE(_offscreen_renderer, m) {
    bindGLEnum(m);

    py::class_<OpenGLContext, std::shared_ptr<OpenGLContext>>(m, "OpenGLContext")
        .def(py::init(&OpenGLContext::construct), py::arg("width"), py::arg("height"))
        .def("resize",      &OpenGLContext::resize, py::arg("width"), py::arg("height"), py::arg("skipViewportCall") = false)
        .def("makeCurrent", &OpenGLContext::makeCurrent)
        .def("finish",      &OpenGLContext::finish)
        .def("buffer",                &OpenGLContext::buffer,                py::return_value_policy::reference)
        .def("unpremultipliedBuffer", &OpenGLContext::unpremultipliedBuffer)
        .def("enable",      [](OpenGLContext &ctx, GLenumWrapper cap) { ctx. enable(unwrapGLenum(cap)); }, py::arg("capability"))
        .def("disable",     [](OpenGLContext &ctx, GLenumWrapper cap) { ctx.disable(unwrapGLenum(cap)); }, py::arg("capability"))
        .def("blendFunc",   [](OpenGLContext &ctx, GLenumWrapper sf, GLenumWrapper df) {
                ctx.blendFunc(unwrapGLenum(sf), unwrapGLenum(df)); }, py::arg("sfactor"), py::arg("dfactor"))
        .def("blendFunc",   [](OpenGLContext &ctx, GLenumWrapper sf, GLenumWrapper df, GLenumWrapper asf, GLenumWrapper adf) {
                ctx.blendFunc(unwrapGLenum( sf), unwrapGLenum( df),
                              unwrapGLenum(asf), unwrapGLenum(adf)); }, py::arg("sfactor"), py::arg("dfactor"), py::arg("alpha_sfactor"), py::arg("alpha_dfactor"))
        .def("cullFace",    [](OpenGLContext &ctx, GLenumWrapper face) {
                ctx.cullFace(unwrapGLenum(face)); }, py::arg("face") = GLenumWrapper::wGL_BACK)
        .def("clear",       &OpenGLContext::clear,    py::arg("color") = Eigen::Vector3f::Zero())
        .def("writePPM",    &OpenGLContext::writePPM, py::arg("path"), py::arg("unpremultiply") = true)
#if PNG_WRITER
        .def("writePNG",    &OpenGLContext::writePNG, py::arg("path"), py::arg("unpremultiply") = true)
#endif
        .def_property_readonly("width",  &OpenGLContext::getWidth)
        .def_property_readonly("height", &OpenGLContext::getHeight)
        ;

    py::class_<Uniform>(m, "Uniform")
        .def_readonly("loc",   &Uniform::loc)
        .def_readonly("size",  &Uniform::size)
        .def_readonly("name",  &Uniform::name)
        .def_property_readonly("type", [](const Uniform &u) { return wrapGLenum(u.type); })
        .def("__repr__", [](const Uniform &u) { return "Uniform '" + u.name + "': " + getGLenumRepr(u.type); })
        ;

    py::class_<Attribute>(m, "Attribute")
        .def_readonly("loc",   &Attribute::loc)
        .def_readonly("size",  &Attribute::size)
        .def_readonly("name",  &Attribute::name)
        .def_property_readonly("type", [](const Attribute &a) { return wrapGLenum(a.type); })
        .def("__repr__", [](const Attribute &a) { return "Attribute " + std::to_string(a.loc) + " ('" + a.name + "'): " + getGLenumRepr(a.type); })
        ;

    py::class_<Shader> pyShader(m, "Shader");
    pyShader
        .def(py::init<std::shared_ptr<OpenGLContext>, const std::string &, const std::string &>(), py::arg("ctx"), py::arg("vtx"), py::arg("frag"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def(py::init<std::shared_ptr<OpenGLContext>, const std::string &, const std::string &, const std::string &>(), py::arg("ctx"), py::arg("vtx"), py::arg("frag"), py::arg("geo"), py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>())
        .def("use", &Shader::use)
        .def_property_readonly("uniforms",   &Shader::getUniforms,   py::return_value_policy::reference)
        .def_property_readonly("attributes", &Shader::getAttributes, py::return_value_policy::reference)
        ;

    MetaMap<BindSetUniform, int, float, Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector4f,
                                        Eigen::Matrix2f, Eigen::Matrix3f, Eigen::Matrix4f>::run(pyShader);

    py::class_<BufferObject>(m, "BufferObject")
        .def("bind", &BufferObject::bind)
        .def("updateData", [](BufferObject &b, Eigen::Ref<const MXfR > data, GLenumWrapper usage) { b.updateData(data, unwrapGLenum(usage)); }, py::arg("data"), py::arg("usage") = GLenumWrapper::wGL_DYNAMIC_DRAW)
        .def("updateData", [](BufferObject &b, Eigen::Ref<const MXuiR> data, GLenumWrapper usage) { b.updateData(data, unwrapGLenum(usage)); }, py::arg("data"), py::arg("usage") = GLenumWrapper::wGL_DYNAMIC_DRAW)
        ;

    py::class_<VertexArrayObject> pyVAO(m, "VertexArrayObject");
    pyVAO
        .def(py::init<std::shared_ptr<OpenGLContext>>(), py::arg("ctx"))
        .def("setAttribute",     &VertexArrayObject::setAttribute,   py::arg("index"), py::arg("A"))
        .def("setIndexBuffer",   &VertexArrayObject::setIndexBuffer, py::arg("A"))
        .def("unsetIndexBuffer", &VertexArrayObject::unsetIndexBuffer)
        .def("bind", &VertexArrayObject::bind)
        .def("draw", &VertexArrayObject::draw, py::arg("shader"))
        .def_property_readonly("attributeBuffers", &VertexArrayObject::attributeBuffers, py::return_value_policy::reference)
        .def_property_readonly("indexBuffer",      &VertexArrayObject::indexBuffer,      py::return_value_policy::reference)
        ;

    MetaMap<BindSetConstAttribute, int, float, Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector4f,
                                               Eigen::Matrix2f, Eigen::Matrix3f, Eigen::Matrix4f>::run(pyVAO);
}
