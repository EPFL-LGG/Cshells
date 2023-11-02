////////////////////////////////////////////////////////////////////////////////
// GLTypeTraits.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Template metaprogramming for converting C/Eigen types to GLSL types.
//  Currently we only convert some of the basic integer and single precision
//  floating point types from:
//      https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/glGetActiveUniform.xhtml
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/22/2020 11:48:27
////////////////////////////////////////////////////////////////////////////////
#ifndef GLTYPES_HH
#define GLTYPES_HH

#include <Eigen/Dense>
#include <GL/glew.h>
#include <type_traits>
#include <map>

inline const char *getGLTypeName(GLenum type) {
    static const std::map<GLenum, const char *> lut {
        {GL_FLOAT       , "GL_FLOAT"       },
        {GL_INT         , "GL_INT"         },
        {GL_UNSIGNED_INT, "GL_UNSIGNED_INT"},
        {GL_BOOL        , "GL_BOOL"        },
        {GL_FLOAT_VEC2  , "GL_FLOAT_VEC2"  },
        {GL_FLOAT_VEC3  , "GL_FLOAT_VEC3"  },
        {GL_FLOAT_VEC4  , "GL_FLOAT_VEC4"  },
        {GL_FLOAT_MAT2  , "GL_FLOAT_MAT2"  },
        {GL_FLOAT_MAT3  , "GL_FLOAT_MAT3"  },
        {GL_FLOAT_MAT4  , "GL_FLOAT_MAT4"  }
    };
    auto it = lut.find(type);
    if (it == lut.end()) throw std::runtime_error("Unhandled type id: " + std::to_string(type));
    return it->second;
}

template<typename T>
struct GLTypeTraitsImpl;

template<> struct GLTypeTraitsImpl<GLfloat>         { static constexpr GLenum type = GL_FLOAT       ; };
template<> struct GLTypeTraitsImpl<GLint>           { static constexpr GLenum type = GL_INT         ; };
template<> struct GLTypeTraitsImpl<GLuint>          { static constexpr GLenum type = GL_UNSIGNED_INT; };
template<> struct GLTypeTraitsImpl<bool>            { static constexpr GLenum type = GL_BOOL        ; };
template<> struct GLTypeTraitsImpl<Eigen::Vector2f> { static constexpr GLenum type = GL_FLOAT_VEC2  ; };
template<> struct GLTypeTraitsImpl<Eigen::Vector3f> { static constexpr GLenum type = GL_FLOAT_VEC3  ; };
template<> struct GLTypeTraitsImpl<Eigen::Vector4f> { static constexpr GLenum type = GL_FLOAT_VEC4  ; };
template<> struct GLTypeTraitsImpl<Eigen::Matrix2f> { static constexpr GLenum type = GL_FLOAT_MAT2  ; };
template<> struct GLTypeTraitsImpl<Eigen::Matrix3f> { static constexpr GLenum type = GL_FLOAT_MAT3  ; };
template<> struct GLTypeTraitsImpl<Eigen::Matrix4f> { static constexpr GLenum type = GL_FLOAT_MAT4  ; };

// "Decay" Eigen expression templates to their underlying evaluated type.
// Leave non-Eigen types unchanged.
template<typename T>
struct EigenDecay { using type = T; };
template<typename Derived>
struct EigenDecay<Eigen::MatrixBase<Derived>> { using type = decltype(std::declval<Derived>().eval()); };

template<typename T>
using GLTypeTraits = GLTypeTraitsImpl<typename EigenDecay<std::decay_t<T>>::type>;

#endif /* end of include guard: GLTYPES_HH */
