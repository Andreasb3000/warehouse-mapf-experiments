#pragma once
#include "common.h"

struct Task {
    int id;
    int loc;
    bool is_parking = false;
    bool movable = false;

    Task() : id(-1), loc(-1) {
    }
    Task(int id, int loc, bool is_parking = false, bool movable = false)
        : id(id), loc(loc), is_parking(is_parking), movable(movable) {
    }
};
