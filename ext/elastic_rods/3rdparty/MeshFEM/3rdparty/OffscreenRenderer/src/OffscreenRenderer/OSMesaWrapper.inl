////////////////////////////////////////////////////////////////////////////////
// OSMesaWrapper.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  A C++ wrapper for RAII OSMesa context creation, rendering, and
//  destruction.
//  We also work around the completely broken implementation of multiple
//  contexts in OSMesa. The following simple example still induces a segfault:
//      https://lists.freedesktop.org/archives/mesa-dev/2016-September/129410.html
//  and generating multiple contexts using my wrappers gives very strange
//  results.
//
//  Our work-around is to use a single OSMesa context (held by a singleton class)
//  with one large buffer and make instances of our derived OSMesaWrapper class
//  draw into and read from sub-rectangles of this buffer.
//  We refer to these instances "virtual contexts".
//  For simplicity, we concatenate the buffers vertically (even though
//  this will be wasteful for virtual contexts with differing widths; a
//  future improvement could employ some more efficient packing strategy.)
//
//  As a side effect, ***adding or deleting contexts will overwrite the
//  contents of the other contexts with undefined values***; these other
//  contexts must be re-rendered before calling `finish()` to copy out their
//  image data.
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/22/2020 02:47:20
////////////////////////////////////////////////////////////////////////////////
#ifndef OSMESAWRAPPER_HH
#define OSMESAWRAPPER_HH

#include <GL/gl.h>
#define GLAPI extern // Annoyingly, GLEW undefs this macro, breaking osmesa...
#include <GL/osmesa.h>

struct OSMesaWrapper;

namespace detail {
    struct OSMesaContextSingleton {
        static OSMesaContextSingleton &getInstance() {
            static std::unique_ptr<OSMesaContextSingleton> ctx;
            if (ctx == nullptr) ctx = std::unique_ptr<OSMesaContextSingleton>(new OSMesaContextSingleton); // make_unique cannot call private constructor
            return *ctx;
        }

        std::tuple<int, int, int, int> rectForVirtualContext(const OSMesaWrapper *vctx) const;
        void resizeToFit();

        void makeCurrent(const OSMesaWrapper *vctx) {
            if (!OSMesaMakeCurrent(m_ctx, m_buffer.data(), GL_UNSIGNED_BYTE, m_width, m_height))
                throw std::runtime_error("OSMesaMakeCurrent failed!");

            static bool firstTime = true;
            if (firstTime) {
                // Initialize GLEW entry points for our new context
                GLenum status = glewInit();
                // Silence spurious error with headless EGL
                // https://github.com/nigels-com/glew/issues/273
                if ((status != GLEW_OK) && (status != GLEW_ERROR_NO_GLX_DISPLAY)) {
                    std::cerr << "GLEW Error: " << glewGetErrorString(status) << std::endl;
                    throw std::runtime_error("glewInit failure");
                }
                std::cout << "gl version: " << glGetString(GL_VERSION) << std::endl;
                firstTime = false;
            }

            GLint x, y, w, h;
            std::tie(x, y, w, h) = rectForVirtualContext(vctx);
            // Render to the correct rectangular region of the context's buffer
            glViewport(x, y, w, h); 
            glScissor (x, y, w, h);
            glEnable(GL_SCISSOR_TEST);
        }

        int getWidth()  const { return m_width;  }
        int getHeight() const { return m_height; }

        void addVirtualContext(const OSMesaWrapper *vctx) {
            // std::cout << "Adding virtual context " << vctx << std::endl;
            m_virtualContexts.push_back(vctx);
            resizeToFit();
        }

        void removeVirtualContext(const OSMesaWrapper *vctx) {
            // std::cout << "Removing virtual context " << vctx << std::endl;
            size_t pos = m_virtualContexts.size();
            for (size_t i = 0; i < m_virtualContexts.size(); ++i) {
                if (m_virtualContexts[i] == vctx) {
                    if (pos != m_virtualContexts.size()) throw std::logic_error("Virtual context registered multiple times");
                    pos = i;
                }
            }
            if (pos == m_virtualContexts.size()) throw std::runtime_error("Virtual context is not registered with this OSMesa context");
            m_virtualContexts.erase(m_virtualContexts.begin() + pos);

            resizeToFit();
        }

        bool virtualContextIsRegistered(const OSMesaWrapper *vctx) const {
            for (auto vctx_i : m_virtualContexts)
                if (vctx_i == vctx) return true;
            return false;
        }

        using Image = Eigen::Array<unsigned char, Eigen::Dynamic, Eigen::Dynamic>;
        Eigen::Ref<const Image> imageForVirtualContext(const OSMesaWrapper *vctx) const {
            int x, y, w, h;
            std::tie(x, y, w, h) = rectForVirtualContext(vctx);
            // std::cout << "Accessing " << w << " x " << h << " image at " << x << ", " << y << std::endl;
            Eigen::Map<const Image> fullCanvas(m_buffer.data(), 4 * m_width, m_height);
            return fullCanvas.block(x, y, 4 * w, h);
        }

        // Write out the whole canvas (for debugging)
        void writePNG(const std::string &path) const {
            write_png_RGBA(path, m_width, m_height, m_buffer.data(), /* verticalFlip = */ true);
        }

        ~OSMesaContextSingleton() {
            OSMesaDestroyContext(m_ctx);
        }

    private:
        OSMesaContextSingleton() {
            const GLint ctxAttribs[] = {
                OSMESA_FORMAT,                GLint(OSMESA_RGBA),
                OSMESA_DEPTH_BITS,            GLint(24),
                OSMESA_STENCIL_BITS,          GLint(0),
                OSMESA_ACCUM_BITS,            GLint(0),
                OSMESA_CONTEXT_MAJOR_VERSION, 3,
                OSMESA_CONTEXT_MINOR_VERSION, 1,
                0
            };
            m_ctx = OSMesaCreateContextAttribs(ctxAttribs, /* sharelist = */ NULL);
            if (!m_ctx) throw std::runtime_error("OSMesaCreateContext failed!");
        }

        int m_width = 0, m_height = 0;
        OpenGLContext::ImageBuffer m_buffer;
        std::vector<const OSMesaWrapper *> m_virtualContexts; // virtual contexts will register/remove themselves from this ordered list
        OSMesaContext m_ctx;
    };
}

struct OSMesaWrapper : public OpenGLContext {
    OSMesaWrapper(int width, int height) {
        resize(width, height);
        ctx().addVirtualContext(this);
    }

    virtual ~OSMesaWrapper() {
        ctx().removeVirtualContext(this);
    }

    detail::OSMesaContextSingleton &ctx() { return detail::OSMesaContextSingleton::getInstance(); }

private:
    virtual void m_resizeImpl(int /* width */, int /* height */) override {
        // Notify osmesa context of our new size if we are already registered
        // (but not if this call is triggered by the OSMesaWrapper constructor).
        if (ctx().virtualContextIsRegistered(this)) {
            ctx().resizeToFit();
        }
    }

    virtual void m_makeCurrent() override { ctx().makeCurrent(this); }

    virtual void m_readImage() {
        Eigen::Map<detail::OSMesaContextSingleton::Image>(m_buffer.data(), m_width * 4, m_height) = ctx().imageForVirtualContext(this);
    }
};

////////////////////////////////////////////////////////////////////////////////
// Implementation of OSMesaContextSingleton methods that need access to
// OSMesaWrapper as a non-incomplete type.
////////////////////////////////////////////////////////////////////////////////
namespace detail{
    std::tuple<int, int, int, int> OSMesaContextSingleton::rectForVirtualContext(const OSMesaWrapper *vctx) const {
        int x = 0, y = 0;
        for (auto vctx_i : m_virtualContexts) {
            if (vctx_i == vctx)
                return std::make_tuple(x, y, vctx->getWidth(), vctx->getHeight());
            y += vctx_i->getHeight();
        }
        throw std::runtime_error("Virtual context is not registered with this OSMesa context");
    }

    void OSMesaContextSingleton::resizeToFit() {
        if (m_virtualContexts.size() == 0) return;
        m_width = 0;
        m_height = 0;
        for (auto vctx_i : m_virtualContexts) {
            m_height += vctx_i->getHeight();
            m_width   = std::max(m_width, vctx_i->getWidth());
        }
        m_buffer.resize(m_width * m_height * 4);
        makeCurrent(m_virtualContexts.back()); // side-effect: apply the updated size to the OSMesa context.
        // std::cout << "Resized to " << m_width << " x " << m_height << std::endl;
    }
}


#endif /* end of include guard: OSMESAWRAPPER_HH */
