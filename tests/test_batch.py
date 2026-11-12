import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from av_common.exports import write_ply
from batch.queue import run

class BatchTests(unittest.TestCase):
    def test_resumable_batch(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); points=np.random.default_rng(2).uniform(size=(200,3))
            fields={name:points[:,i] for i,name in enumerate('xyz')}
            for name in ['red','green','blue']: fields[name]=np.full(200,128,dtype='uint8')
            write_ply(d/'source.ply',fields)
            manifest=d/'jobs.json'; manifest.write_text(json.dumps({'schema_version':1,'jobs':[{'id':'scene','input':'source.ply','output':'scene.spz','config':{'voxel_size':.01}}]}))
            self.assertEqual(run(manifest,d/'state.json')[0]['status'],'done')
            self.assertEqual(run(manifest,d/'state.json')[0]['status'],'skipped')

    def test_missing_input_is_recorded_and_other_jobs_continue(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); points=np.random.default_rng(5).uniform(size=(300,3))
            fields={name:points[:,i] for i,name in enumerate('xyz')}
            for name in ['red','green','blue']: fields[name]=np.full(len(points),128,dtype='uint8')
            write_ply(d/'valid.ply',fields)
            manifest=d/'jobs.json'; manifest.write_text(json.dumps({'schema_version':1,'jobs':[{'id':'missing','input':'missing.ply','output':'missing-output.ply','config':{}},{'id':'valid','input':'valid.ply','output':'valid-output.ply','config':{}}]}))
            state=d/'state.json'; outcomes=run(manifest,state)
            self.assertEqual([item['status'] for item in outcomes],['failed','done'])
            self.assertEqual(json.loads(state.read_text())['jobs']['missing']['status'],'failed')
            self.assertFalse((d/'missing-output.ply').exists()); self.assertTrue((d/'valid-output.ply').exists())
