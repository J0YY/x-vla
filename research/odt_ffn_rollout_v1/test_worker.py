"""Pure worker-protocol regressions without simulator construction."""
import unittest
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
from research.odt_ffn_rollout_v1.worker import ARMS,projection,array_digest,task_authority,offline_gate,validate_observation,validated_transition,state_vector,rgb_image
from research.odt_ffn_rollout_v1.guards import source_audit,install
from research.odt_ffn_native_v1.common import digest

source_audit()
install()


def observation_fixture():
    return {"agentview_image":np.zeros((64,64,3),dtype=np.uint8),"robot0_eef_pos":np.zeros(3,dtype=np.float64),
        "robot0_eef_quat":np.array([0.,0.,0.,1.],dtype=np.float64),"robot0_gripper_qpos":np.zeros(2,dtype=np.float64)}


class WorkerTests(unittest.TestCase):
    def test_smoke_projection_conservative_full_horizon(self):
        records=[{"arm":arm,"measurement_complete":True,"setup_elapsed_s":1.,"elapsed_s":10.,"steps":280} for arm in ARMS]
        measured=projection(records)
        self.assertTrue(measured["passed"])
        self.assertEqual(measured["projected_serial80_episode_seconds"],1100.)
        records[0]["measurement_complete"]=False
        self.assertFalse(projection(records)["passed"])
    def test_slow_or_incomplete_smoke_never_admits_full_pilot(self):
        records=[{"arm":arm,"measurement_complete":True,"setup_elapsed_s":1.,"elapsed_s":100.,"steps":280} for arm in ARMS]
        self.assertFalse(projection(records)["passed"])
        self.assertFalse(projection(records[:3])["passed"])
    def test_state_hash_binds_dtype_shape_and_values(self):
        values=np.arange(4,dtype=np.float64)
        self.assertNotEqual(array_digest(values),array_digest(values.astype(np.float32)))
        self.assertNotEqual(array_digest(values),array_digest(values.reshape(2,2)))
    def test_offline_gate_rejects_v1_and_missing_failure_accounting_before_arrays(self):
        for schema,accounting in (("odt-ffn-native-actions-v1",2),("odt-ffn-native-actions-v2",1)):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder)
                manifest={"workflow_completed":True,"schema":schema,"failure_accounting_version":accounting}
                (root/"manifest.json").write_text(json.dumps(manifest))
                args=argparse.Namespace(offline=root,offline_sha256=digest(root/"manifest.json"),variants=root/"missing")
                with self.assertRaisesRegex(ValueError,"workflow"):
                    offline_gate(args)
    def test_offline_gate_rejects_v2_source_drift_before_publication_or_arrays(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            manifest={"workflow_completed":True,"schema":"odt-ffn-native-actions-v2","failure_accounting_version":2,"source_sha256":{}}
            (root/"manifest.json").write_text(json.dumps(manifest))
            args=argparse.Namespace(offline=root,offline_sha256=digest(root/"manifest.json"),variants=root/"missing")
            with self.assertRaisesRegex(ValueError,"source closure"):
                offline_gate(args)
    def test_transition_accepts_only_finite_real_scalar_and_literal_boolean(self):
        observation=observation_fixture()
        for reward in (0.,np.float64(1.),np.float32(.5),1):
            _,actual,done=validated_transition((observation,reward,np.bool_(False),{}),"test")
            self.assertEqual(actual,float(reward));self.assertIs(done,False)
        for reward in (np.nan,np.inf,-np.inf,np.array(0.),np.array([0.]),True,"0",None):
            with self.assertRaisesRegex(RuntimeError,"reward"):
                validated_transition((observation,reward,False,{}),"settle step 0")
        for done in (0,1,"false",np.array(False),None):
            with self.assertRaisesRegex(RuntimeError,"done"):
                validated_transition((observation,0.,done,{}),"policy step 0")
    def test_transition_tuple_schema_and_observation_mapping_required(self):
        observation=observation_fixture()
        for value in ([observation,0.,False,{}],(observation,0.,False),None):
            with self.assertRaisesRegex(RuntimeError,"schema"):validated_transition(value,"test")
        with self.assertRaisesRegex(RuntimeError,"mapping"):
            validated_transition((None,0.,False,{}),"test")
    def test_observation_missing_shapes_dtypes_nonfinite_fail_before_conversion(self):
        valid=observation_fixture()
        for field in valid:
            missing=valid.copy();del missing[field]
            with self.assertRaisesRegex(RuntimeError,"omits"):validate_observation(missing)
            wrong=valid.copy();wrong[field]=valid[field].reshape(-1)[:1]
            with self.assertRaisesRegex(RuntimeError,"shape"):validate_observation(wrong)
            wrong=valid.copy();wrong[field]=valid[field].astype(np.float32)
            with self.assertRaisesRegex(RuntimeError,"dtype"):validate_observation(wrong)
        for field in ("robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos"):
            wrong=valid.copy();wrong[field]=valid[field].copy();wrong[field][0]=np.nan
            with self.assertRaisesRegex(RuntimeError,"finiteness"):validate_observation(wrong)
        wrong=valid.copy();wrong["robot0_gripper_qpos"]=np.zeros(3,dtype=np.float64)
        with self.assertRaisesRegex(RuntimeError,"shape"):state_vector(wrong)
        wrong=valid.copy();wrong["agentview_image"]=np.zeros((128,128,3),dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError,"shape"):rgb_image(wrong)
    def test_task_authority_and_source_closure(self):
        authority=task_authority()["EXPECTED_TASK_PROTOCOL"]
        self.assertEqual(len(authority),10)
        self.assertIn("alphabet soup",authority["0"][0])
        audit=source_audit()
        self.assertEqual(audit["local"]["controller.py"],"e89c2b9efe953086859a5c9f37fcbca0fef1fcf9dd9f8c301fe94ce61aaf2687")
        self.assertFalse(any("modal_odt_dimension_curve_worker" in path for path in audit["native_source_audit"]["source_sha256"]))


if __name__=="__main__":unittest.main()
