////////////////////////////////////////////////////////////////////////////////
// Buffers.hh
////////////////////////////////////////////////////////////////////////////////
/*! @file
//  Manage VAOs, VBOs
*/
//  Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
//  Created:  09/23/2020 17:40:22
////////////////////////////////////////////////////////////////////////////////
#ifndef BUFFERS_HH
#define BUFFERS_HH

#include "GLTypeTraits.hh"
#include "RAIIGLResource.hh"
#include "UASetters.hh"

// We need to use row-major types in order to interpret each row as giving a
// vertex's attributes (following libigl conventions).
using MXfR  = Eigen::Matrix<float       , Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MXuiR = Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct BufferObject : RAIIGLResource<BufferObject> {
    using Base = RAIIGLResource<BufferObject>;
    using Base::id;

    BufferObject() : Base(std::weak_ptr<OpenGLContext>()) { } // Allow creation of a dummy, unallocated buffer object not tied to any context
    BufferObject(std::weak_ptr<OpenGLContext> ctx, Eigen::Ref<const MXfR > A) : Base(ctx) { glGenBuffers(1, &id); this->m_validateConstruction(); updateData(A); }
    BufferObject(std::weak_ptr<OpenGLContext> ctx, Eigen::Ref<const MXuiR> A) : Base(ctx) { glGenBuffers(1, &id); this->m_validateConstruction(); updateData(A); }

    void bind(GLenum target) const { glBindBuffer(target, id); }

    template<class Derived>
    void updateData(const Eigen::MatrixBase<Derived> &A,
                    GLenum usage = GL_DYNAMIC_DRAW) { // In our typical use cases, buffers may change every frame...
        bind(GL_ARRAY_BUFFER);
        glBufferData(GL_ARRAY_BUFFER,
                     A.size() * sizeof(typename Derived::Scalar),
                     A.derived().data(),
                     usage); // We assume buffers change rarely
        m_count = A.rows();
    }

    size_t count() const { return m_count; }

private:
    friend struct RAIIGLResource<BufferObject>;
    void m_delete() { glDeleteBuffers(1, &id); }
    size_t m_count = 0;
};

struct VertexArrayObject : RAIIGLResource<VertexArrayObject> {
    using Base = RAIIGLResource<VertexArrayObject>;
    using Base::id;

    VertexArrayObject(std::weak_ptr<OpenGLContext> ctx) : Base(ctx) {
        glGenVertexArrays(1, &id);
        this->m_validateConstruction();
    }

    // Create/update a buffer with data for attribute `loc`.
    // Each row of A is interpreted as a vertex attribute,
    // so A's column size determines the attribute size.
    void setAttribute(int loc, Eigen::Ref<const MXfR> A) {
        bind();
        auto it = m_attributes.find(loc);
        if (it == m_attributes.end()) it = m_attributes.emplace(loc, BufferObject(this->m_ctx, A)).first;
        else {
            if (!it->second.allocated()) it->second = BufferObject(this->m_ctx, A); // former attribute at `loc` was a dummy/empty; we must allocate a real one.
            else                         it->second.updateData(A);
        }
        const auto &buf = it->second;
        buf.bind(GL_ARRAY_BUFFER);
        glVertexAttribPointer(loc,
                              A.cols(), GL_FLOAT, // # cols floats per vertex
                              GL_FALSE,  // Don't normalize
                              0, 0); // No gap between vertex data, and no offset from array beginning.
        glCheckError();
        glEnableVertexAttribArray(loc);
        glCheckError();
    }

    template<typename T>
    void setConstantAttribute(int loc, const T &a) {
        // WARNING: the generic value set here is global and *not*
        // part of the VAO's state. Therefore this method should
        // be called before each draw of the VAO to make sure someone
        // else didn't change the value...
        if (m_attributes.count(loc) == 0) {
            // Create a dummy, empty attribute to satisfy validation in `draw`
            m_attributes[loc] = BufferObject(); 
        }

        bind();
        glDisableVertexAttribArray(GLuint(loc));
        detail::setAttribute(GLuint(loc), a);

        glCheckError();
    }

    void setIndexBuffer(Eigen::Ref<const MXuiR> A) {
        bind();
        glCheckError();
        Eigen::Map<const Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>> flatA(A.data(), A.size());
        if (m_indexBuffer.allocated()) m_indexBuffer.updateData(flatA);
        else  m_indexBuffer = BufferObject(this->m_ctx, flatA);
        m_indexBuffer.bind(GL_ELEMENT_ARRAY_BUFFER);
        glCheckError();
    }

    void unsetIndexBuffer() {
        bind();
        m_indexBuffer = BufferObject();
        glCheckError();
    }

    void bind() const { glBindVertexArray(id); }

    void draw(const Shader &s) const {
        size_t numChecked = 0;
        for (const auto &attr : s.getAttributes()) {
            if (m_attributes.count(attr.loc)) ++numChecked;
            else if (!attr.isBuiltIn) { // Ignore auto-generated attributes like gl_VertexID
                throw std::runtime_error("Attribute " + std::to_string(attr.loc) + " (" + attr.name + ") is not set in VAO");
            }
        }
        if (numChecked != m_attributes.size()) throw std::runtime_error("Extraneous attributes found in VAO");
        if (!s.allUniformsSet()) {
            std::string msg = "Unset uniform(s):";
            for (const Uniform &u : s.getUniforms())
                msg += " " + u.name;
            throw std::runtime_error(msg);
        }

        s.use();
        glCheckError();

        bind();
        glCheckError();

        if (m_indexBuffer.allocated()) {
            // std::cout << "glDrawElements (indexed)" << std::endl;
            glDrawElements(GL_TRIANGLES, m_indexBuffer.count(), GL_UNSIGNED_INT, NULL);
        }
        else {
            // std::cout << "glDrawArrays (unindexed)" << std::endl;
            glDrawArrays(GL_TRIANGLES, 0, m_attributes.at(0).count());
        }
        glCheckError();
    }

    const std::map<int, BufferObject> &attributeBuffers() const { return m_attributes;  }
    const BufferObject                &indexBuffer()      const { return m_indexBuffer; }

private:
    std::map<int, BufferObject> m_attributes;
    BufferObject m_indexBuffer;
    friend struct RAIIGLResource<VertexArrayObject>;
    void m_delete() { glDeleteVertexArrays(1, &id); /* std::cout << "Delete vertex array " << id << std::endl; */ }
};

#endif /* end of include guard: BUFFERS_HH */
