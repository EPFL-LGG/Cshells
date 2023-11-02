#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <stdexcept>
#include "OpenGLContext.hh"
#include "Shader.hh"
#include "Buffers.hh"

struct RenderState {
    RenderState(int width, int height) {
        ctx = OpenGLContext::construct(width, height);
        ctx->makeCurrent();
        shader = Shader::fromFiles(ctx, SHADER_PATH "/demo.vert", SHADER_PATH "/demo.frag");
        for (const auto &u : shader->getUniforms()) {
            std::cout << "Uniform " << u.loc << ": " << u.name << std::endl;
        }
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> positions;
        positions << -0.5f, -0.5f, 0.0f,
                      0.5f, -0.5f, 0.0f,
                      -0.5f,  0.5f, 0.0f;
        Eigen::Matrix<float, 3, 4, Eigen::RowMajor> colors;
        colors << 1.0f, 0.0f, 0.0f, 1.0f,
                  0.0f, 1.0f, 0.0f, 1.0f,
                  0.0f, 0.0f, 1.0f, 1.0f;
        Eigen::Matrix<unsigned int, 3, 1> indices;
        indices << 0, 1, 2;
        vao = std::make_unique<VertexArrayObject>(ctx);
        vao->setAttribute(0, positions);
        vao->setAttribute(1, colors);
        vao->setIndexBuffer(indices);
    }

    void render() {
        ctx->render([&]() {
            // glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glEnable(GL_DEPTH_TEST);

            vao->draw(*shader);
        });

        ctx->finish();
    }

    std::shared_ptr<OpenGLContext> ctx;
    std::unique_ptr<VertexArrayObject> vao;
    std::unique_ptr<Shader> shader;
};

int main(int argc, char *argv[])
{
    int width = 400;
    int height = 400;
    char *filename = NULL;

    if (argc < 2) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "  osdemo filename [width height]\n");
        return 0;
    }

    filename = argv[1];
    if (argc == 4) {
        width = atoi(argv[2]);
        height = atoi(argv[3]);
    }

    auto render1 = std::make_unique<RenderState>(width, height);
    auto render2 = std::make_unique<RenderState>(2 * width, 2 * height);
    render1->render();
    render1->ctx->writePNG(filename + std::string("1.png"));

    render2->render();
    render2->ctx->writePNG(filename + std::string("2.png"));

    render1->render();
    render2->ctx->writePNG(filename + std::string("3.png")); // Make sure context 2's buffer is not affected!

    // Test deletion
    render2.reset();
    render1->render(); // this is essential since for our OSMesa virtual contexts, removing one leaves the others' buffers undefined!
    render1->ctx->writePNG(filename + std::string("4.png"));

    return 0;
}
