////////////////////////////////////////////////////////////////////////////////
// Shader.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Basic RAII wrappers for shader programs.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/22/2020 11:26:37
////////////////////////////////////////////////////////////////////////////////
#ifndef SHADER_HH
#define SHADER_HH

#include <vector>
#include <array>
#include <stdexcept>
#include <fstream>
#include <sstream>

#include "GLTypeTraits.hh"
#include "GLErrors.hh"
#include "RAIIGLResource.hh"
#include "UASetters.hh"

// A vertex/fragment/geometry/etc shader object that can be compiled and linked
// into a program
struct ShaderObject : RAIIGLResource<ShaderObject> {
    using Base = RAIIGLResource<ShaderObject>;
    using Base::id;

    // Create shader and load in source (but do not compile yet!)
    ShaderObject(std::weak_ptr<OpenGLContext> ctx, const std::string &source, GLenum type)
        : Base(ctx, glCreateShader(type))
    {
        if (source.size() == 0) throw std::runtime_error("Empty shader");
        const char *src = source.c_str();
        glShaderSource(id, /* one string */1, &src, /* source is null terminated */ NULL);
        glCheckError("shader construction");
    }

    void compile() {
        glCompileShader(id);
        glCheckStatus(id, GL_COMPILE_STATUS);
        glCheckError("shader compilation");
    }
private:
    friend struct RAIIGLResource<ShaderObject>;
    void m_delete() { glDeleteShader(id); /* std::cout << "Deleted shader " << id << std::endl; */ }
};

struct Uniform {
    Uniform(GLuint prog, GLuint index) {
        std::array<char, 512> buf;
        glGetActiveUniform(prog, index, 512, NULL, &size, &type, buf.data());
        name = std::string(buf.data());
        loc = glGetUniformLocation(prog, buf.data()); // Can differ from index!!!
        if (loc < 0) throw std::logic_error("Couldn't look up uniform location");
    }

    template<typename T>
    void set(const T &val) {
        using Traits = GLTypeTraits<T>;
        if (Traits::type != type) throw std::runtime_error("Uniform type mismatch");
        detail::setUniform(loc, val);
        isSet = true;
    }

    GLint loc;
    GLint size;
    GLenum type;
    std::string name;
    bool isSet = false;
};

struct Attribute {
    Attribute(GLuint prog, GLuint index) {
        std::array<char, 512> buf;
        glGetActiveAttrib(prog, index, 512, NULL, &size, &type, buf.data());
        name = std::string(buf.data());

        loc = glGetAttribLocation(prog, buf.data()); // Can differ from index!!!
        isBuiltIn = name.substr(0, 3) == "gl_"; // Some drivers apparently don't assign built-in attributes like gl_VertexID valid locations
        if (!isBuiltIn && (loc < 0)) throw std::logic_error("Couldn't look up attribute location for " + name);
    }

    GLint loc;
    GLint size;
    GLenum type;
    std::string name;
    bool isBuiltIn;
};

// Resource for a GLSL Shader
struct Shader {
    struct Program : RAIIGLResource<Program> {
        using Base = RAIIGLResource<Program>;
        using Base::id;
        Program(std::weak_ptr<OpenGLContext> ctx) : Base(ctx, glCreateProgram()) { this->m_validateConstruction(); }

    private:
        friend struct RAIIGLResource<Program>;
        void m_delete() { glDeleteProgram(id); /* std::cout << "Deleted program " << id << std::endl; */ }
    };

    using Sources = std::vector<std::string>;
    Shader(std::weak_ptr<OpenGLContext> ctx,
           const Sources &vtxSources,
           const Sources &fragSources,
           const Sources &geoSources = Sources())
        : m_prog(ctx)
    {
        for (const auto &s :  vtxSources) m_objects.emplace_back(ctx, s, GL_VERTEX_SHADER);
        for (const auto &s : fragSources) m_objects.emplace_back(ctx, s, GL_FRAGMENT_SHADER);
        for (const auto &s :  geoSources) m_objects.emplace_back(ctx, s, GL_GEOMETRY_SHADER);

        // Compile and attach all shader objects
        for (auto &obj : m_objects) {
            obj.compile();
            glAttachShader(m_prog.id, obj.id);
            glCheckError("attach shader");
        }

        glLinkProgram(m_prog.id);
        glCheckStatus(m_prog.id, GL_LINK_STATUS);

        // Get uniform information
        GLint numUniforms;
        glGetProgramiv(m_prog.id, GL_ACTIVE_UNIFORMS, &numUniforms);
        for (GLuint i = 0; i < GLuint(numUniforms); ++i)
            m_uniforms.emplace_back(m_prog.id, i);

        GLint numAttributes;
        glGetProgramiv(m_prog.id, GL_ACTIVE_ATTRIBUTES, &numAttributes);
        for (GLuint i = 0; i < GLuint(numAttributes); ++i)
            m_attributes.emplace_back(m_prog.id, i);
    }

    Shader(std::weak_ptr<OpenGLContext> ctx, const std::string &vtx, const std::string &frag, const std::string &geo)
        : Shader(ctx, Sources(1, vtx), Sources(1, frag), Sources(1, geo)) { }

    Shader(std::weak_ptr<OpenGLContext> ctx, const std::string &vtx, const std::string &frag)
        : Shader(ctx, Sources(1, vtx), Sources(1, frag)) { }

    // Copy is disallowed, but move is ok.
    Shader(const Shader &) = delete;
    Shader(Shader &&b) :
        m_prog(std::move(b.m_prog)),
        m_objects(std::move(b.m_objects)),
        m_uniforms(std::move(b.m_uniforms)),
        m_attributes(std::move(b.m_attributes))
    { }

    static std::string readFile(const std::string &path) {
        std::ifstream inFile(path);
        if (!inFile.is_open()) throw std::runtime_error("Couldn't open input file " + path);
        std::stringstream ss;
        ss << inFile.rdbuf();
        return ss.str();
    }

    // Factory method to construct from shaders stored in files.
    static std::unique_ptr<Shader> fromFiles(std::weak_ptr<OpenGLContext> ctx, const std::string &vtxFile, const std::string &fragFile, const std::string &geoFile = std::string()) {
        if (geoFile.size())
            return std::make_unique<Shader>(ctx, readFile(vtxFile), readFile(fragFile), readFile(geoFile));
        return std::make_unique<Shader>(ctx, readFile(vtxFile), readFile(fragFile));
    }

    void use() const { glUseProgram(m_prog.id); }

    template<typename T>
    void setUniform(const std::string &name, const T &val) {
        for (Uniform &u : m_uniforms) {
            if (u.name == name) {
                use();
                glCheckError("pre setUniform");
                u.set(val);
                glCheckError("setUniform");
                return;
            }
        }
        throw std::runtime_error("Uniform not present: " + name);
    }

    bool allUniformsSet() const {
        for (const Uniform &u : m_uniforms)
            if (!u.isSet) return false;
        return true;
    }

    const std::vector<Uniform>   &getUniforms  () const { return m_uniforms; }
    const std::vector<Attribute> &getAttributes() const { return m_attributes; }
private:
    Program m_prog;
    std::vector<ShaderObject> m_objects;
    std::vector<Uniform>      m_uniforms;
    std::vector<Attribute>    m_attributes;
};

#endif /* end of include guard: SHADER_HH */
