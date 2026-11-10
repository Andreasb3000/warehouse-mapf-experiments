#include "utils/safety_geometry.h"
#include <cassert>
#include <iostream>

int main() {
    const safety::Rectangle polygon{1.2, 0.2, 0.3};
    const safety::Pose origin{0, 0, 0};
    // Boundary footprint intersection, not just centre-point containment.
    assert(safety::intersectsDisc(polygon, origin, {1.27, 0}, 0.085));
    assert(!safety::intersectsDisc(polygon, origin, {1.30, 0}, 0.085));
    assert(!safety::intersectsDisc(polygon, origin, {0.8, 0.5}, 0.085));
    assert(safety::intersectsDisc(polygon, origin, {-0.25, 0}, 0.085));
    assert(!safety::intersectsDisc(polygon, origin, {-0.30, 0}, 0.085));
    // Orientation in the ARGoS world frame; 90 degrees faces world +y.
    assert(safety::intersectsDisc(polygon, {10, 20, M_PI / 2}, {10, 21}, 0.085));
    assert(!safety::intersectsDisc(polygon, {10, 20, M_PI / 2}, {11, 20}, 0.085));
    // Corner near-miss rejects an inflated-AABB false positive.
    assert(!safety::intersectsDisc(polygon, origin, {1.28, 0.38}, 0.085));
    assert(safety::intersectsBox(polygon, origin, {{1.5, 0, 0}, .4, .4}));
    assert(!safety::intersectsBox(polygon, origin, {{1.7, 0, 0}, .4, .4}));
    assert(safety::intersectsBox(polygon, origin, {{1.6, 0, M_PI / 4}, .4, .4}));
    assert(!safety::intersectsBox(polygon, origin, {{0, 1, 0}, .4, .4}));
    // Long wall segments can cross a region despite their centre being far away.
    assert(safety::intersectsBox(polygon, origin, {{0.8, 0, M_PI / 2}, 10, .025}));
    assert(safety::intersectsBox(polygon, {4, 7, M_PI / 2}, {{4, 8.5, M_PI / 2}, .4, .4}));
    // Head-on fields meet well before either body reaches the other's field.
    const safety::Pose head_on{2.2, 0, M_PI};
    assert(safety::intersectsRectangles(polygon, origin, polygon, head_on));
    assert(safety::intersectsRectangles(polygon, head_on, polygon, origin));
    assert(!safety::intersectsDisc(polygon, origin, {head_on.x, head_on.y}, .085));
    assert(!safety::intersectsDisc(polygon, head_on, {0, 0}, .085));
    // Following and crossing fields can overlap while the bodies stay outside.
    const safety::Pose following{1.35, 0, 0}, crossing{1.0, -1.0, M_PI / 2};
    assert(safety::intersectsRectangles(polygon, origin, polygon, following));
    assert(!safety::intersectsDisc(polygon, origin, {1.35, 0}, .085));
    assert(safety::intersectsRectangles(polygon, origin, polygon, crossing));
    assert(!safety::intersectsDisc(polygon, origin, {1.0, -1.0}, .085));
    assert(!safety::intersectsDisc(polygon, crossing, {0, 0}, .085));
    // Parallel edge contact counts; an actual gap does not.
    assert(safety::intersectsRectangles(polygon, origin, polygon, {0, .6, 0}));
    assert(!safety::intersectsRectangles(polygon, origin, polygon, {0, .6001, 0}));
    // Each robot contributes its own asymmetric field and orientation.
    assert(!safety::intersectsRectangles(polygon, origin, {.2, .2, .1}, {1.5, 0, 0}));
    assert(safety::intersectsRectangles(polygon, origin, {.2, .5, .1}, {1.5, 0, 0}));
    assert(safety::intersectsRectangles(polygon, {4, 7, M_PI / 2}, polygon,
                                        {4, 9.2, -M_PI / 2}));
    std::cout << "28 safety geometry assertions passed\n";
}
