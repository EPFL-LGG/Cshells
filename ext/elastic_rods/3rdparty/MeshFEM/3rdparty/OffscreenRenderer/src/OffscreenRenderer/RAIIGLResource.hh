////////////////////////////////////////////////////////////////////////////////
// RAIIGLResource.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  CRTP class providing a safe wrapper around an OpenGL resource (shader,
//  buffer, etc.) guarding against leaks, dangling ids, and double frees.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/23/2020 17:37:35
////////////////////////////////////////////////////////////////////////////////
#ifndef RAIIGLRESOURCE_HH
#define RAIIGLRESOURCE_HH

#include "OpenGLContext.hh"

// Resource linked to a particular context
template<class Derived>
struct RAIIGLResource {
    RAIIGLResource(std::weak_ptr<OpenGLContext> ctx) : m_ctx(ctx) { }
    RAIIGLResource(std::weak_ptr<OpenGLContext> ctx, GLuint _id) : id(_id), m_ctx(ctx) {
        m_validateConstruction();
    }

    // Eliminate dangerous copy constructor/assignment;
    // provide move constructor/assignment instead.
    RAIIGLResource(const RAIIGLResource &) = delete;
    RAIIGLResource(RAIIGLResource &&b) : id(b.id) { b.id = 0; }

    RAIIGLResource &operator=(const RAIIGLResource &  ) = delete;
    RAIIGLResource &operator=(      RAIIGLResource &&b) { id = b.id; b.id = 0; m_ctx = b.m_ctx; b.m_ctx.reset(); return *this; }

    // Note: if a context is destroyed, the driver should automatically
    // deallocate all of its resources (assuming they are not shared by
    // another context).
    bool allocated() const { return (id != 0) && !m_ctx.expired(); }

    ~RAIIGLResource() {
        if (allocated()) {
            // std::cout << "Deleting resource " << id << std::endl;
            auto ctx = m_ctx.lock();
            if (!ctx) { std::cerr << "WARNING: could not lock context" << std::endl; return; }
            ctx->makeCurrent();
            static_cast<Derived *>(this)->m_delete();
        }
    }

    GLuint id = 0;
protected:
    std::weak_ptr<OpenGLContext> m_ctx;
    void m_validateConstruction() {
        glCheckError("resource creation");
        if (id == 0) throw std::runtime_error("Resource creation failed");
    }
};

#endif /* end of include guard: RAIIGLRESOURCE_HH */
