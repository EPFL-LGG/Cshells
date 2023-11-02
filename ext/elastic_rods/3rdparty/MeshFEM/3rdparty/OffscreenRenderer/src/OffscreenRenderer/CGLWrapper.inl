////////////////////////////////////////////////////////////////////////////////
// CGLWrapper.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Wrapper for RAII CGL context creation, rendering, and destruction.
//  Adapted from
//      https://stackoverflow.com/questions/37077935/osx-offscreen-rendering-coregl-framebuffers-shaders-headache
//      http://renderingpipeline.com/2012/05/windowless-opengl-on-macos-x/
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/22/2020 05:15:43
////////////////////////////////////////////////////////////////////////////////
#ifndef EGLWRAPPER_HH
#define EGLWRAPPER_HH

#include <OpenGL/CGLTypes.h>
#include <OpenGL/CGLCurrent.h>
#include <OpenGL/OpenGL.h>
#include "GLErrors.hh"

struct CGLWrapper : public OpenGLContext {
    CGLWrapper(int width, int height, GLenum /* format */ = GL_RGBA,
               GLint depthBits = 24, GLint stencilBits = 0, GLint accumBits = 0)
    {
        // Initialize CGL
        CGLPixelFormatAttribute pixAttributes[12] = {
            kCGLPFAAccelerated,   // Presence of boolean flags implies `true` value according to documentation
            kCGLPFAOpenGLProfile, CGLPixelFormatAttribute(kCGLOGLPVersion_3_2_Core), // Uses latest available version (not actually 3.2!!)
            kCGLPFADepthSize,     CGLPixelFormatAttribute(depthBits),
            kCGLPFAAlphaSize,     CGLPixelFormatAttribute(8),
            kCGLPFAStencilSize,   CGLPixelFormatAttribute(stencilBits),
            kCGLPFAAccumSize,     CGLPixelFormatAttribute(accumBits),
            CGLPixelFormatAttribute(0)
        };

        {
            CGLPixelFormatObj pix;
            GLint num;
            CGLError errorCode = CGLChoosePixelFormat(pixAttributes, &pix, &num);
            if (errorCode != kCGLNoError) throw std::runtime_error("CGLChoosePixelFormat failure");

            errorCode = CGLCreateContext(pix, NULL, &m_ctx);
            if (errorCode != kCGLNoError) throw std::runtime_error("CGLCreateContext failure");

            CGLDestroyPixelFormat(pix);
        }

        // std::cout << "Initialize CGL " << m_ctx << std::endl;

        m_makeCurrent();

        std::cout << glGetString(GL_VERSION) << std::endl;

        // Initialize GLEW entry points for our new context
        m_glewInit();

        glGenFramebuffers(1,  &m_frameBufferID);
        glGenRenderbuffers(1, &m_renderBufferID);
        glGenRenderbuffers(1, &m_depthBufferID);

        resize(width, height); // Trigger buffer generation/binding
    }

    virtual ~CGLWrapper() {
        // std::cout << "Destroy CGL " << m_ctx << std::endl;
        makeCurrent();

        auto oldErrors = glGetErrorString(); // Flush any old errors
        if (oldErrors.size())
            std::cerr << "Unreported errors found on context destruction:" << std::endl << oldErrors << std::endl;

        glDeleteRenderbuffers(1, &m_renderBufferID);
        glDeleteRenderbuffers(1, &m_depthBufferID);
        glDeleteFramebuffers (1, &m_frameBufferID);

        // CGLSetCurrentContext(nullptr);
        CGLDestroyContext(m_ctx);
    }

private:
    virtual void m_makeCurrent() override {
        // std::cout << "Make current CGL " << m_ctx << std::endl;
        CGLError errorCode = CGLSetCurrentContext(m_ctx);
        if (errorCode != kCGLNoError)
            throw std::runtime_error("CGLSetCurrentContext failure");
    }

    virtual void m_readImage() override {
        glBindFramebuffer(GL_FRAMEBUFFER, m_frameBufferID);
        glBindRenderbuffer(GL_RENDERBUFFER, m_renderBufferID);
        glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, m_buffer.data());
        glCheckError("Read image");
    }

    virtual void m_resizeImpl(int width, int height) override {
        glBindRenderbuffer(GL_RENDERBUFFER, m_renderBufferID);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, width, height);

        glBindRenderbuffer(GL_RENDERBUFFER, m_depthBufferID);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);

        glBindFramebuffer(GL_FRAMEBUFFER, m_frameBufferID);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_renderBufferID);
        glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT , GL_RENDERBUFFER,  m_depthBufferID);

        glCheckError("allocate framebuffers");

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            throw std::runtime_error("framebuffer is not complete!");
    }

    CGLContextObj m_ctx = nullptr;
    GLuint m_frameBufferID = 0,
           m_renderBufferID = 0,
           m_depthBufferID = 0;
};

#endif /* end of include guard: EGLWRAPPER_HH */
