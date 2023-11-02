#version 140
#extension GL_ARB_explicit_attrib_location : enable

layout (location = 0) in vec3 v_position; // bind v_position to attribute 0
layout (location = 1) in vec4 v_color;    // bind v_color    to attribute 1

out vec4 v4f_color; // Interpolated output to be read by fragment shader

void main() {
    // pass through color and position from the vertex attributes
    v4f_color = v_color;
    gl_Position = vec4(v_position, 1.0);
}
