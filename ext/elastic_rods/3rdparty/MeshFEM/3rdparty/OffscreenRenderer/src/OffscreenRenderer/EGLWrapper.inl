////////////////////////////////////////////////////////////////////////////////
// EGLWrapper.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Wrapper for RAII EGL context creation, rendering, and destruction.
//  Adapted from https://developer.nvidia.com/blog/egl-eye-opengl-visualization-without-x-server/
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/22/2020 05:15:43
////////////////////////////////////////////////////////////////////////////////
#ifndef EGLWRAPPER_HH
#define EGLWRAPPER_HH

#include <EGL/egl.h>

namespace detail {
    // All contexts share an EGL display...
    struct EGLDisplaySingleton {
        static EGLDisplaySingleton &getInstance() {
            static std::unique_ptr<EGLDisplaySingleton> display;
            if (display == nullptr) display = std::unique_ptr<EGLDisplaySingleton>(new EGLDisplaySingleton); // make_unique cannot call private constructor
            return *display;
        }

        EGLDisplay get() const { return m_display; }

        ~EGLDisplaySingleton() {
            eglTerminate(m_display);
        }
    private:
        EGLDisplay m_display;

        EGLDisplaySingleton() {
            // Initialize EGL
            m_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);

            EGLint major, minor;
            eglInitialize(m_display, &major, &minor);
            // std::cout << major << ", " << minor << std::endl;

            // Bind the API
            eglBindAPI(EGL_OPENGL_API);
        }
	};
}

struct EGLWrapper : public OpenGLContext {
    EGLWrapper(int width, int height, GLenum /* format */ = GL_RGBA,
               GLint depthBits = 24, GLint /* stencilBits */ = 0, GLint /* accumBits */ = 0)
        : m_display(detail::EGLDisplaySingleton::getInstance())
    {
        // Select an appropriate configuration
        EGLint numConfigs;

        const EGLint configAttribs[] = {
            EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
            EGL_BLUE_SIZE, 8,
            EGL_GREEN_SIZE, 8,
            EGL_RED_SIZE, 8,
            EGL_ALPHA_SIZE, 8,
            EGL_DEPTH_SIZE, depthBits,
            EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
            EGL_NONE
        };

        eglChooseConfig(m_display.get(), configAttribs, &m_config, 1, &numConfigs);

        resize(width, height);
        makeCurrent();
        // std::cout << "gl version: " << glGetString(GL_VERSION) << std::endl;
    }

    virtual ~EGLWrapper() {
        m_destroy_size_specific();
    }

private:
    virtual void m_makeCurrent() override {
        eglMakeCurrent(m_display.get(), m_surf, m_surf, m_ctx);
		EGLint lastError = eglGetError();
		if (lastError != EGL_SUCCESS)
			std::cerr << "Error " << lastError << " in eglMakeCurrent" << std::endl;
        glCheckError();
        // std::cout << "Made current " << this << std::endl;
    }

    void m_destroy_size_specific() {
        if (m_ctx  != nullptr) { eglDestroyContext(m_display.get(), m_ctx ); m_ctx  = nullptr; }
        if (m_surf != nullptr) { eglDestroySurface(m_display.get(), m_surf); m_surf = nullptr; }
    }

    virtual void m_readImage() override {
        glReadPixels(0, 0, m_width, m_height, GL_RGBA, GL_UNSIGNED_BYTE, m_buffer.data());
    }

    virtual void m_resizeImpl(int width, int height) override {
        m_destroy_size_specific();

        const EGLint pbufferAttribs[] = {
            EGL_WIDTH, width,
            EGL_HEIGHT, height,
            EGL_NONE,
        };

        // Create a surface
        m_surf = eglCreatePbufferSurface(m_display.get(), m_config, pbufferAttribs);
        if (!m_surf) throw std::runtime_error("eglCreatePbufferSurface failed");

        // Create a 3.3 context and make it current
        EGLint contextAttribs[] = {
            EGL_CONTEXT_MAJOR_VERSION, 3,
            EGL_CONTEXT_MINOR_VERSION, 3,
            EGL_NONE
        };
        m_ctx = eglCreateContext(m_display.get(), m_config, EGL_NO_CONTEXT, contextAttribs);
        if (!m_surf) throw std::runtime_error("eglCreateContext failed");

        EGLint version;
        eglQueryContext(m_display.get(), m_ctx, EGL_CONTEXT_CLIENT_VERSION, &version);
        std::cout << "Created EGL context with version " << version << std::endl;

        m_makeCurrent();

        // Initialize GLEW entry points for our new context
        m_glewInit();
    }

    const detail::EGLDisplaySingleton &m_display;
    EGLConfig  m_config  = nullptr;
    EGLSurface m_surf    = nullptr;
    EGLContext m_ctx     = nullptr;
};

#endif /* end of include guard: EGLWRAPPER_HH */
