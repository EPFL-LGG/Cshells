//
//
// 2013 (c) Mathieu Courtemanche

#ifndef VISVALINGAM_ALGORITHM_H
#define VISVALINGAM_ALGORITHM_H

#include <vector>
#include <cassert>
#include <cstddef>

namespace visvalingam_simplify {


using VertexIndex = size_t;

struct Point
{
    Point() {}
    Point(double inX, double inY) : X(inX), Y(inY) {}

    double X;
    double Y;
};

using Linestring = std::vector<Point>;
using MultiLinestring = std::vector<Linestring>;

struct Polygon
{
    Polygon() {}

    Linestring exterior_ring;
    MultiLinestring interior_rings;
};

using MultiPolygon = std::vector<Polygon>;

// returns cross product between two vectors: v1 ^ v2 in right handed coordinate
// E.g.: returned value on +z axis
double cross_product(const Point& v1, const Point& v2);

// A - B
Point vector_sub(const Point& A, const Point& B);

class Visvalingam_Algorithm
{
public:
    Visvalingam_Algorithm(const Linestring& input);

    void simplify(double area_threshold, Linestring* res) const;
    void simplify(size_t numVertices, Linestring &result) const;

    void print_areas() const;

private:
    bool contains_vertex(VertexIndex vertex_index, double area_threshold) const;

    std::vector<double> m_effective_areas;
    const Linestring& m_input_line;
};

inline bool
Visvalingam_Algorithm::contains_vertex(VertexIndex vertex_index,
                                        double area_threshold) const
{
    assert(vertex_index < m_effective_areas.size());
    assert(m_effective_areas.size() != 0);
    if (vertex_index == 0 || vertex_index == m_effective_areas.size()-1)
    {
        // end points always kept since we don't evaluate their effective areas
        return true;
    }
    return m_effective_areas[vertex_index] > area_threshold;
}

}

#endif // VISVALINGAM_ALGORITHM_H
