enum class GLenumWrapper : GLenum {
    // Types
    wGL_FLOAT        = GL_FLOAT       ,
    wGL_INT          = GL_INT         ,
    wGL_UNSIGNED_INT = GL_UNSIGNED_INT,
    wGL_BOOL         = GL_BOOL        ,
    wGL_FLOAT_VEC2   = GL_FLOAT_VEC2  ,
    wGL_FLOAT_VEC3   = GL_FLOAT_VEC3  ,
    wGL_FLOAT_VEC4   = GL_FLOAT_VEC4  ,
    wGL_FLOAT_MAT2   = GL_FLOAT_MAT2  ,
    wGL_FLOAT_MAT3   = GL_FLOAT_MAT3  ,
    wGL_FLOAT_MAT4   = GL_FLOAT_MAT4  ,

    // Constants for glBlendFunc
    wGL_ONE                 = GL_ONE,
    wGL_ZERO                = GL_ZERO,
    wGL_SRC_ALPHA           = GL_SRC_ALPHA,
    wGL_ONE_MINUS_SRC_ALPHA = GL_ONE_MINUS_SRC_ALPHA,

    // Culling mode constants
    wGL_FRONT          = GL_FRONT,
    wGL_BACK           = GL_BACK,
    wGL_FRONT_AND_BACK = GL_FRONT_AND_BACK,

    // Capabilities
    wGL_DEPTH_TEST = GL_DEPTH_TEST,
    wGL_BLEND      = GL_BLEND,
    wGL_CULL_FACE  = GL_CULL_FACE,

    // Buffer usage
    wGL_DYNAMIC_DRAW = GL_DYNAMIC_DRAW,
    wGL_STATIC_DRAW  = GL_STATIC_DRAW
};

GLenumWrapper wrapGLenum(GLenum val) {
    return static_cast<GLenumWrapper>(val);
}

GLenum unwrapGLenum(GLenumWrapper val) {
    return static_cast<GLenum>(val);
}

// Get a string representation of the GLenum "val"
// (Only works for the values for which we have created Python bindings)
std::string getGLenumRepr(GLenum val) {
    return std::string(py::repr(py::cast(wrapGLenum(val))));
}

void bindGLEnum(py::module &m) {
    py::enum_<GLenumWrapper>(m, "GLenum")
        // Types
        .value("GL_FLOAT"       , GLenumWrapper::wGL_FLOAT)
        .value("GL_INT"         , GLenumWrapper::wGL_INT)
        .value("GL_UNSIGNED_INT", GLenumWrapper::wGL_UNSIGNED_INT)
        .value("GL_BOOL"        , GLenumWrapper::wGL_BOOL)
        .value("GL_FLOAT_VEC2"  , GLenumWrapper::wGL_FLOAT_VEC2)
        .value("GL_FLOAT_VEC3"  , GLenumWrapper::wGL_FLOAT_VEC3)
        .value("GL_FLOAT_VEC4"  , GLenumWrapper::wGL_FLOAT_VEC4)
        .value("GL_FLOAT_MAT2"  , GLenumWrapper::wGL_FLOAT_MAT2)
        .value("GL_FLOAT_MAT3"  , GLenumWrapper::wGL_FLOAT_MAT3)
        .value("GL_FLOAT_MAT4"  , GLenumWrapper::wGL_FLOAT_MAT4)

        // Constants for glBlendFunc
        .value("GL_ONE",                 GLenumWrapper::wGL_ONE)
        .value("GL_ZERO",                GLenumWrapper::wGL_ZERO)
        .value("GL_SRC_ALPHA",           GLenumWrapper::wGL_SRC_ALPHA)
        .value("GL_ONE_MINUS_SRC_ALPHA", GLenumWrapper::wGL_ONE_MINUS_SRC_ALPHA)

        // Culling mode constants
        .value("GL_FRONT",          GLenumWrapper::wGL_FRONT)
        .value("GL_BACK",           GLenumWrapper::wGL_BACK)
        .value("GL_FRONT_AND_BACK", GLenumWrapper::wGL_FRONT_AND_BACK)

        // Capabilities
        .value("GL_DEPTH_TEST", GLenumWrapper::wGL_DEPTH_TEST)
        .value("GL_BLEND",      GLenumWrapper::wGL_BLEND)
        .value("GL_CULL_FACE",  GLenumWrapper::wGL_CULL_FACE)

        // Buffer usage
        .value("GL_DYNAMIC_DRAW", GLenumWrapper::wGL_DYNAMIC_DRAW)
        .value("GL_STATIC_DRAW",  GLenumWrapper::wGL_STATIC_DRAW)
        ;
}
