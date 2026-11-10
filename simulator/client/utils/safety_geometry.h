#ifndef LSMART_SAFETY_GEOMETRY_H
#define LSMART_SAFETY_GEOMETRY_H

#include <algorithm>
#include <array>
#include <cmath>

// Geometry for the experimental ideal local sensor. Distances are metres.
// A rectangle extends front/back along the robot's local +x/-x axes.
namespace safety {
struct Point { double x, y; };
struct Pose { double x, y, yaw; };
struct Rectangle { double front, back, half_width; };
struct Box { Pose pose; double half_x, half_y; };

inline Point local(Point world, Pose origin) {
    const double dx = world.x - origin.x, dy = world.y - origin.y;
    return {std::cos(origin.yaw) * dx + std::sin(origin.yaw) * dy,
            -std::sin(origin.yaw) * dx + std::cos(origin.yaw) * dy};
}

inline bool intersectsDisc(Rectangle rectangle, Pose origin, Point centre,
                           double radius) {
    const auto p = local(centre, origin);
    const double dx = p.x - std::clamp(p.x, -rectangle.back, rectangle.front);
    const double dy = p.y - std::clamp(p.y, -rectangle.half_width, rectangle.half_width);
    return dx * dx + dy * dy <= radius * radius;
}

inline std::array<Point, 4> corners(Box box) {
    std::array<Point, 4> points;
    const double c = std::cos(box.pose.yaw), s = std::sin(box.pose.yaw);
    int i = 0;
    for (const auto p : {Point{-box.half_x, -box.half_y},
                         Point{box.half_x, -box.half_y},
                         Point{box.half_x, box.half_y},
                         Point{-box.half_x, box.half_y}}) {
        points[i++] = {box.pose.x + c * p.x - s * p.y,
                       box.pose.y + s * p.x + c * p.y};
    }
    return points;
}

inline Box asBox(Rectangle rectangle, Pose origin) {
    const double shift = (rectangle.front - rectangle.back) / 2;
    return {{origin.x + shift * std::cos(origin.yaw),
             origin.y + shift * std::sin(origin.yaw), origin.yaw},
            (rectangle.front + rectangle.back) / 2, rectangle.half_width};
}

inline bool intersectsBox(Rectangle rectangle, Pose origin, Box box) {
    const Box region = asBox(rectangle, origin);
    const auto a = corners(region), b = corners(box);
    for (const double yaw : {origin.yaw, box.pose.yaw}) {
        for (const auto axis : {Point{std::cos(yaw), std::sin(yaw)},
                                Point{-std::sin(yaw), std::cos(yaw)}}) {
            double amin = INFINITY, amax = -INFINITY;
            double bmin = INFINITY, bmax = -INFINITY;
            for (const auto p : a) {
                const auto v = p.x * axis.x + p.y * axis.y;
                amin = std::min(amin, v); amax = std::max(amax, v);
            }
            for (const auto p : b) {
                const auto v = p.x * axis.x + p.y * axis.y;
                bmin = std::min(bmin, v); bmax = std::max(bmax, v);
            }
            if (amax < bmin || bmax < amin) return false;
        }
    }
    return true;
}

// Closed rectangles: edge/corner contact counts as intersection. Both fields
// follow their own robot's heading and may have different dimensions.
inline bool intersectsRectangles(Rectangle first, Pose first_pose,
                                 Rectangle second, Pose second_pose) {
    return intersectsBox(first, first_pose, asBox(second, second_pose));
}
}  // namespace safety
#endif
