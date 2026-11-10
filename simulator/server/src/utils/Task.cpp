#include "utils/Task.h"

std::ostream& operator<<(std::ostream& os, const Task& task) {
    os << "( loc : " << task.location << ", ori : " << task.orientation
       << ", wait_time : " << task.task_wait_time
       << ", hold_time : " << task.hold_time
       << ", is_parking : " << task.is_parking << ", movable : " << task.movable
       << ", id : " << task.id << " )";
    return os;
}

vector<list<Task>> read_task_vec(const std::string& fname, int num_of_drives) {
    std::ifstream input(fname);
    if (!input) throw std::runtime_error("Cannot open task file: " + fname);

    auto read_line = [&input](string &line) {
        while (std::getline(input, line)) {
            boost::trim(line);
            if (line.empty() || line[0] != '#') return true;
        }
        return false;
    };
    auto parse_integer = [&fname](string value) {
        boost::trim(value);
        try {
            size_t end = 0;
            int result = std::stoi(value, &end);
            if (end != value.size()) throw std::invalid_argument("trailing data");
            return result;
        } catch (const std::exception &) {
            throw std::runtime_error("Invalid integer in task file " + fname + ": " + value);
        }
    };
    string line;
    if (!read_line(line) || parse_integer(line) != num_of_drives) {
        throw std::runtime_error("Task file robot count must match the simulation.");
    }
    vector<list<Task>> task_vec;
    for (int i = 0; i < num_of_drives; i++) {
        if (!read_line(line)) {
            throw std::runtime_error("Task file has fewer robot rows than declared.");
        }
        list<Task> task_list;
        vector<string> raw_tasks;
        boost::split(raw_tasks, line, boost::is_any_of(";"));
        for (auto raw_task : raw_tasks) {
            boost::trim(raw_task);
            if (raw_task.empty()) continue;
            vector<string> fields;
            boost::split(fields, raw_task, boost::is_any_of(","));
            if (fields.size() != 4) {
                throw std::runtime_error("Each task must be (location,orientation,wait,movable).");
            }
            int loc = parse_integer(fields[0]);
            int ori = parse_integer(fields[1]);
            int wait = parse_integer(fields[2]);
            int movable = parse_integer(fields[3]);
            if (wait < 0 || (movable != 0 && movable != 1)) {
                throw std::runtime_error("Invalid wait or movable field in task file.");
            }
            task_list.emplace_back(loc, ori, wait, 0, -1, -1, false, movable != 0);
        }
        task_vec.push_back(task_list);
    }
    while (read_line(line)) {
        if (!line.empty()) throw std::runtime_error("Task file has extra robot rows.");
    }
    return task_vec;
}

vector<tuple<int, int>> read_start_vec(const std::string& fname,
                                       int num_of_drives) {
    auto raw_int_vec = read_int_vec(fname, num_of_drives);
    vector<tuple<int, int>> start_vec;  // vector of (loc, ori)
    for (int i = 0; i < num_of_drives; i++) {
        auto start_tuple =
            make_tuple(raw_int_vec[i].front(), raw_int_vec[i].back());
        start_vec.push_back(start_tuple);
    }
    return start_vec;
}

void from_json(const json& j, Task& t) {
    j.at("location").get_to(t.location);
    j.at("orientation").get_to(t.orientation);
    j.at("task_wait_time").get_to(t.task_wait_time);
    j.at("hold_time").get_to(t.hold_time);
    j.at("agent_id").get_to(t.agent_id);
    j.at("finish_t").get_to(t.finish_t);
    j.at("is_parking").get_to(t.is_parking);
    j.at("movable").get_to(t.movable);
    j.at("id").get_to(t.id);
}

void to_json(json& j, const Task& t) {
    j = {
        {"location", t.location},
        {"orientation", t.orientation},
        {"task_wait_time", t.task_wait_time},
        {"hold_time", t.hold_time},
        {"agent_id", t.agent_id},
        {"finish_t", t.finish_t},
        {"is_parking", t.is_parking},
        {"movable", t.movable},
        {"id", t.id},
    };
}
