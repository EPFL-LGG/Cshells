# OffscreenRenderer
A GPU-accelerated headless renderer with Python bindings.

`OffscreenRenderer` supports accelerated OpenGL rendering to offscreen
contexts on macOS and Linux created using the standard platform-specific APIs
(CGL on macOS, EGL on Linux).
An unaccelerated OSMesa-based code path is provided for CPU-based rendering,
but it requires a version of OSMesa supporting OpenGL 3.1 and an
OSMesa-compatible version of `GLEW`; I have failed to get this working so far.

The built-in shaders support transparency (with automatic depth sorting) and anti-aliased
wireframe rendering based on an improved version of [Single-pass Wireframe
Rendering](https://dl.acm.org/doi/abs/10.1145/1179849.1180035) that does not
need a geometry shader.

We also support rendering directly to a video using `ffmpeg`.

Please see `python/MeshDemo.ipynb` for a demonstration of the Python bindings
and `src/OffscreenRenderer/demo.cc` for a simple C++ example.
