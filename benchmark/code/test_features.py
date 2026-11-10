import unittest
from copy import deepcopy
import numpy as np
from features import rasterize,window_mean,interpolate,observation_clock
from local_model import local_features


class FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.scenario={'agents':[{'id':0,'goals':[[8,8]]}]}

    def test_stationary_robot_is_slow_only_with_complete_history(self):
        x=np.zeros((31,1,3));x[:,:,:2]=3
        r,e=rasterize(x,self.scenario,(10,10))
        self.assertEqual(float(r[:20,1].sum()),0)
        self.assertTrue(e[20:].all())
        self.assertEqual(float(r[:,0].sum()),31)

    def test_turns_service_and_fast_motion_excluded(self):
        for kind in ('turn','service','fast'):
            x=np.zeros((31,1,3));x[:,:,:2]=3
            if kind=='turn':x[:,0,2]=np.arange(31)*.05
            if kind=='service':x[:,:,:2]=8
            if kind=='fast':x[:,0,0]=3+np.arange(31)*.1
            _,e=rasterize(x,self.scenario,(10,10))
            self.assertFalse(e.any(),kind)

    def test_future_cannot_change_historical_features(self):
        x=np.zeros((101,1,3));x[:,:,:2]=3
        before,_=rasterize(x,self.scenario,(10,10))
        x[60:,:,0]=7
        after,_=rasterize(x,self.scenario,(10,10))
        np.testing.assert_array_equal(before[:60],after[:60])

    def test_nonoverlap_windows_and_units(self):
        x=np.zeros((201,2,2,2));x[:50]=1;x[50:100]=2;x[100:200]=3;x[200]=999
        result=window_mean(x)
        np.testing.assert_array_equal(result[:,0,0,0],[1,2,3])

    def test_yaw_interpolation_wraps_short_way(self):
        states=np.array([[[0,0,np.pi-.1]],[[0,1,-np.pi+.1]]])
        middle=interpolate(np.array([0,1]),states,[.5])
        self.assertAlmostEqual(middle[0,0,2],np.pi)

    def test_local_features_do_not_depend_on_identity_or_mutate_inputs(self):
        hist=np.zeros((6,2,10,10));plan=np.zeros((3,2,10,10));static=np.zeros((3,10,10))
        hist[-1,0,4,4]=1;copy=hist.copy()
        f=local_features(hist,plan,static)
        self.assertEqual(f.shape,(100,42))
        np.testing.assert_array_equal(hist,copy)


class OriginBoundaryTest(unittest.TestCase):
    def test_origin_retains_full_two_second_history(self):
        from features import predicted_rasters
        # At0.31m/s the robot is just above the0.30m/s slow threshold.
        # Duplicating its origin drops one0.1s movement segment, incorrectly
        # reducing the apparent two-second speed to0.2945m/s.
        scenario={'agents':[{'id':0,'goals':[[18,18]]}]}
        native=np.zeros((221,1,3))
        native[:,0,0]=3+np.arange(221)*.031
        native[:,0,1]=4
        past=native[:21].copy()
        forecast={'frames':[{'offset_s':k*.1,'agents':[{
            'id':0,'row':float(native[k+20,0,0]),'col':4.,'yaw':0.}]} for k in range(201)]}
        predicted,_=predicted_rasters(forecast,past,scenario,(20,20))
        actual,_=rasterize(native,scenario,(20,20))
        np.testing.assert_array_equal(predicted,actual[20:])
        self.assertEqual(float(predicted[0,1].sum()),0.)
        duplicated=np.concatenate([past[-20:],native[20:]])
        faulty,_=rasterize(duplicated,scenario,(20,20))
        self.assertEqual(float(faulty[20,1].sum()),1.)


class ClockPhaseTest(unittest.TestCase):
    def event(self):
        return {'clock':130, 'pose_frame':{'clock':130},
                'snapshot':{'clock':130,'server_clock':129},
                'controllers':[{'clock':130,'control_step_count':129,'poll_phase':9}]}

    def test_passive_origin_uses_physics_clock_not_preceding_server_tick(self):
        event=self.event(); original=deepcopy(event)
        self.assertEqual(observation_clock(event),130)
        self.assertEqual(event,original)

    def test_live_handshake_clock_cannot_shift_physical_history_or_poll_phase(self):
        event=self.event();event.update(clock=129,observation_clock=130)
        original=deepcopy(event)
        self.assertEqual(observation_clock(event),130)
        self.assertEqual(event['controllers'][0]['poll_phase'],9)
        self.assertEqual(event,original)

    def test_mixed_pose_and_controller_clocks_are_rejected(self):
        event=self.event();event['controllers'][0]['clock']=129
        with self.assertRaises(ValueError):observation_clock(event)


if __name__=='__main__':unittest.main()
