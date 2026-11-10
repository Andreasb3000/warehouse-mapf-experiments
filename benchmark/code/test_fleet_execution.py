import json
from pathlib import Path
import tempfile
import unittest

from fleet_execution import SAFETY_DEFAULTS, summarize, write_json


class FleetRecordingTest(unittest.TestCase):
    def fixture(self, root, count):
        write_json(root/'scenario.json', {'agents':[{'id':i} for i in range(count)],
                    'safety':{'enabled': True, 'inner_stop_enabled': False, 'robot_trigger':'outer_overlap'}})
        write_json(root/'config.json', {'parameters':{'num_agents':count,'sim_duration':2,'ticks_per_second':10}})
        write_json(root/'map.json', {'layout':['...']})
        write_json(root/'tasks.json', {'tasks':[{'robot_id':i,'stages':[
            {'operation':'pickup','location':[i,0]}, {'operation':'dropoff','location':[i,1]}]} for i in range(count)]})
        events=[{'type':'safety_config','robots':[{**SAFETY_DEFAULTS, 'agent_id':i,'robot_trigger':'outer_overlap',
                 'inner_stop_enabled':False,'enabled':True} for i in range(count)]},
                {'type':'routing_barrier_complete','clock':0}, {'type':'routing_plan','success':True}]
        events += [{'type':'tick','clock':t,'agents':[{'id':i,'x':float(i),'y':t*.1} for i in range(count)]} for t in range(2)]
        events += [{'type':'goal_reached','clock':1,'agent':0,'row':0,'col':c} for c in (0,1)]
        return events

    def save(self, root, events):
        (root/'simulator.log').write_text('\n'.join('native: '+json.dumps(e) for e in events)+'\n')

    def test_real_larger_fleet_count_is_used_everywhere(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); events=self.fixture(root,81); self.save(root,events)
            result=summarize(root)
            self.assertEqual(result['num_agents'],81)
            self.assertEqual(result['task_count'],81)
            self.assertEqual(result['agents_moved'],81)
            self.assertEqual(result['deliveries_completed'],1)
            self.assertEqual(len(result['task_outcomes']),81)

    def test_unfinished_tasks_do_not_invalidate_raw_recording(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); events=self.fixture(root,54); self.save(root,events[:-2])
            result=summarize(root)
            self.assertEqual(result['deliveries_completed'],0)
            self.assertEqual(result['native_frames'],2)

    def test_wrong_robot_set_or_missing_frame_is_rejected(self):
        for change in ('ids','clock'):
            with tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary); events=self.fixture(root,54)
                if change=='ids':events[3]['agents'].pop()
                else:events[4]['clock']=2
                self.save(root,events)
                with self.assertRaises(ValueError):summarize(root)

    def test_wrong_task_acknowledgement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); events=self.fixture(root,54); events[-1]['col']=99; self.save(root,events)
            with self.assertRaises(ValueError):summarize(root)

    def test_native_safety_policy_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); events=self.fixture(root,54); events[0]['robots'][-1]['inner_stop_enabled']=True; self.save(root,events)
            with self.assertRaises(ValueError):summarize(root)

    def test_disabled_agent_responses_are_verified_not_assumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); events=self.fixture(root,2)
            scenario=json.loads((root/'scenario.json').read_text())
            for agent in scenario['agents']:
                agent['safety']={'enabled':False}
            write_json(root/'scenario.json',scenario)
            for robot in events[0]['robots']:
                robot['enabled']=False
            self.save(root,events)
            self.assertEqual(summarize(root)['num_agents'],2)
            events[0]['robots'][0]['enabled']=True
            self.save(root,events)
            with self.assertRaises(ValueError):summarize(root)

    def test_changed_native_field_geometry_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); events=self.fixture(root,2)
            events[0]['robots'][0]['slow_front_m']=2.0
            self.save(root,events)
            with self.assertRaises(ValueError):summarize(root)


if __name__=='__main__':unittest.main()
