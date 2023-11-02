// Improved implementation of "Single-Pass Wireframe Rendering" technique.
// Author:  Julian Panetta (jpanetta), julian.panetta@gmail.com
#version 140
#extension GL_ARB_explicit_attrib_location : enable

// Vertex attributes
layout (location = 0) in vec3  v_position; // bind v_position to attribute 0
layout (location = 1) in vec3  v_normal;   // bind v_normal   to attribute 1
layout (location = 2) in vec4  v_color;    // bind v_color    to attribute 2
layout (location = 3) in float v_height;   // Height of this vertex above the opposte edge (needed for wireframe)

// Transformation matrices
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

// Vertex shader outputs
out vec3 v2f_eyePos;
out vec3 v2f_eyeNormal;
out vec4 v2f_color;

// For drawing wireframe
noperspective out vec3 v2f_barycentric; // Barycentric coordinate functions.

void main() {
    vec4 eyePos = modelViewMatrix * vec4(v_position, 1.0);
    v2f_eyePos    = vec3(eyePos);
    v2f_eyeNormal = normalMatrix * v_normal;
    v2f_color     = v_color;

    v2f_barycentric = vec3(0.0);
    v2f_barycentric[gl_VertexID % 3] = 1.0;

    gl_Position = projectionMatrix * eyePos;
}
