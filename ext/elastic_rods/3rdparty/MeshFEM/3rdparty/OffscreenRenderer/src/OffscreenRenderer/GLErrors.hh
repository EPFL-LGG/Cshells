////////////////////////////////////////////////////////////////////////////////
// GLErrors.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  OpenGL error checking and reporting.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/22/2020 12:40:50
////////////////////////////////////////////////////////////////////////////////
#ifndef GLERRORS_HH
#define GLERRORS_HH

#include <map>
#include <string>
#include <iostream>
#include <GL/glew.h>

inline std::string glGetErrorString() {
    GLenum error;
    std::string result;
    std::map<GLuint, const char *> errorDescriptions = {
            {GL_INVALID_ENUM,                  "Invalid enum"},
            {GL_INVALID_VALUE,                 "Invalid value (out of range)"},
            {GL_INVALID_OPERATION,             "Invalid operation (not allowed in current state)"},
            {GL_INVALID_FRAMEBUFFER_OPERATION, "Invalid framebuffer operation (framebuffer not complete)"},
            {GL_OUT_OF_MEMORY,                 "Out of memory"},
            {GL_STACK_UNDERFLOW,               "Stack underflow"},
            {GL_STACK_OVERFLOW,                "Stack overflow"},
    };
    while ((error = glGetError()) != GL_NO_ERROR) {
        if (result.size()) result += '\n';
        try         { result += errorDescriptions.at(error); }
        catch (...) { result += "Unknown"; }
    }
    return result;
}

inline void glCheckError(const std::string &operation = "") {
    auto err = glGetErrorString();
    if (err.size()) {
        throw std::runtime_error("GL error" + ((operation.size() ? " encountered in " : "") + operation) + ":\n" + err);
    }
}

inline void glCheckStatus(GLint id, GLenum statusType) {
    GLint success;
    std::string desc;
    if (statusType == GL_COMPILE_STATUS) {
        glGetShaderiv(id, GL_COMPILE_STATUS, &success);
        desc = "Shader compilation";
    }
    else if (statusType == GL_LINK_STATUS) {
        glGetProgramiv(id, GL_LINK_STATUS, &success);
        desc = "Program link";
    }
    else {
        throw std::logic_error("Unknown status type");
    }

    if (success == GL_FALSE) {
        char infoLog[512];
        glGetShaderInfoLog(id, 512, NULL, infoLog);
        std::cerr << desc << " failed: " << infoLog << std::endl;
        throw std::runtime_error(desc + " failed -- check cerr");
    }
}

#endif /* end of include guard: GLERRORS_HH */
