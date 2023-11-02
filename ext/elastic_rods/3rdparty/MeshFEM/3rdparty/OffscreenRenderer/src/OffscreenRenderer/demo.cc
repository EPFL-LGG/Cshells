#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Dense>
#include <stdexcept>
#include "OpenGLContext.hh"
#include "Shader.hh"
#include "Buffers.hh"

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

    auto ctx = OpenGLContext::construct(width, height);
    ctx->makeCurrent();

    auto shader = Shader::fromFiles(ctx, SHADER_PATH "/demo.vert", SHADER_PATH "/demo.frag");
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

    VertexArrayObject vao(ctx);
    vao.setAttribute(0, positions);
    vao.setAttribute(1, colors);
    vao.setIndexBuffer(indices);

    ctx->render([&]() {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        vao.draw(*shader);
    });

    ctx->finish();

    if (filename)
        ctx->writePNG(filename);

    return 0;
}
