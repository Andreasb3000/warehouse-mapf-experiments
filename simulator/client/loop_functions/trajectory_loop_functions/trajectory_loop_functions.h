#ifndef TRAJECTORY_LOOP_FUNCTIONS_H
#define TRAJECTORY_LOOP_FUNCTIONS_H

#include <argos3/core/simulator/loop_functions.h>
#include <argos3/plugins/robots/foot-bot/simulator/footbot_entity.h>
#include <argos3/plugins/simulator/visualizations/qt-opengl/qtopengl_render.h>

#include "controllers/footbot_diffusion/footbot_diffusion.h"
#include "utils/common.h"
#include "utils/safety_geometry.h"
#include <argos3/plugins/simulator/entities/box_entity.h>

using namespace argos;

class CTrajectoryLoopFunctions : public CLoopFunctions {
public:
    typedef map<CFootBotEntity*, vector<CVector3> > TWaypointMap;
    TWaypointMap m_tWaypoints;

public:
    virtual ~CTrajectoryLoopFunctions() {
    }

    virtual void Init(TConfigurationNode& t_tree);

    virtual void Reset();

    virtual void PostStep();

    virtual void PreStep();

    void PostExperiment() override;

    bool IsExperimentFinished() override;

    void Destroy() override;

    inline const TWaypointMap& GetWaypoints() const {
        return m_tWaypoints;
    }

private:
    void addMobileVisualization();
    void initSafety();
    void updateSafety();
    void reportSafety();
    void recordPredictionSnapshot();
    json predictionSnapshot();
    void writeRoutingRequest();
    bool record_prediction_snapshots_ = false;

    struct SafetyRobot {
        CFootBotEntity* entity;
        CFootBotDiffusion* controller;
        safety::Rectangle slow, stop;
        double scale, radius;
        bool enabled, inner_stop_enabled, outer_overlap;
        int mode = -1;
        unsigned long long slow_ticks = 0, stop_ticks = 0;
        unsigned long long robot_outer_trigger_ticks = 0, obstacle_outer_trigger_ticks = 0;
        unsigned long long polygon_only_trigger_ticks = 0;
        string blocker_id, trigger_reason;
    };
    vector<SafetyRobot> safety_robots_;
    vector<CBoxEntity*> safety_boxes_;
    bool safety_enabled_ = false;
    bool safety_reported_ = false;

public:
    vector<CVector3> task_stations;

private:
    bool is_initialized = false;
    int port_number = -1;
    shared_ptr<rpc::client> client;
    int time_step_tick = 0;
    bool end_sim = false;
};

#endif
