////////////////////////////////////////////////////////////////////////////////
// UASetters.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Overloads for setting various types of uniforms and attributes
//  (These are low-level unchecked functions; type checking is peformed by the
//  `Uniform` and `Attribute` classes)
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/24/2020 16:45:31
////////////////////////////////////////////////////////////////////////////////
#ifndef UASETTERS_HH
#define UASETTERS_HH

#include <GL/glew.h>

namespace detail {

inline void setUniform(GLint loc,                  GLfloat f) { glUniform1f       (loc,           f);                   }
inline void setUniform(GLint loc,                    GLint i) { glUniform1i       (loc,           i);                   }
inline void setUniform(GLint loc,                   GLuint u) { glUniform1ui      (loc,           u);                   }
inline void setUniform(GLint loc,                     bool b) { glUniform1i       (loc,           static_cast<int>(b)); }
inline void setUniform(GLint loc, const Eigen::Vector2f &vec) { glUniform2fv      (loc, 1,        vec.data());          }
inline void setUniform(GLint loc, const Eigen::Vector3f &vec) { glUniform3fv      (loc, 1,        vec.data());          }
inline void setUniform(GLint loc, const Eigen::Vector4f &vec) { glUniform4fv      (loc, 1,        vec.data());          }
inline void setUniform(GLint loc, const Eigen::Matrix2f &mat) { glUniformMatrix2fv(loc, 1, false, mat.data());          }
inline void setUniform(GLint loc, const Eigen::Matrix3f &mat) { glUniformMatrix3fv(loc, 1, false, mat.data());          }
inline void setUniform(GLint loc, const Eigen::Matrix4f &mat) { glUniformMatrix4fv(loc, 1, false, mat.data());          }

// Constant vertex attribute setters (to be used when glDisableVertexAttribArray is called)
// Note: matrix-valued attributes are accessed as sequential column vector attributes.
inline void setAttribute(GLuint index,                  GLfloat f) { glVertexAttrib1f  (index, f);                   }
inline void setAttribute(GLuint index,                    GLint i) { glVertexAttribI1i (index, i);                   }
inline void setAttribute(GLuint index,                   GLuint u) { glVertexAttribI1ui(index, u);                   }
inline void setAttribute(GLuint index,                     bool b) { glVertexAttribI1i (index, static_cast<int>(b)); }
inline void setAttribute(GLuint index, const Eigen::Vector2f &vec) { glVertexAttrib2fv (index, vec.data());          }
inline void setAttribute(GLuint index, const Eigen::Vector3f &vec) { glVertexAttrib3fv (index, vec.data());          }
inline void setAttribute(GLuint index, const Eigen::Vector4f &vec) { glVertexAttrib4fv (index, vec.data());          }
inline void setAttribute(GLuint index, const Eigen::Matrix2f &mat) { glVertexAttrib2fv (index, mat.col(0).data()); glVertexAttrib2fv(index + 1, mat.col(1).data());                                                                                                   }
inline void setAttribute(GLuint index, const Eigen::Matrix3f &mat) { glVertexAttrib3fv (index, mat.col(0).data()); glVertexAttrib3fv(index + 1, mat.col(1).data()); glVertexAttrib3fv(index + 2, mat.col(2).data());                                                  }
inline void setAttribute(GLuint index, const Eigen::Matrix4f &mat) { glVertexAttrib4fv (index, mat.col(0).data()); glVertexAttrib4fv(index + 1, mat.col(1).data()); glVertexAttrib4fv(index + 2, mat.col(2).data()); glVertexAttrib4fv(index + 3, mat.col(3).data()); }

}

#endif /* end of include guard: UASETTERS_HH */
