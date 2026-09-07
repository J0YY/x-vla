"""Native-runtime integration tests, guards before all model construction."""
import unittest
import numpy as np
from scripts.odt_direct_only_compliance import install_direct_only_runtime_guard, assert_direct_only_runtime_guard
from research.odt_ffn_native_v2.native import source_audit, graph_module, replacement_block, action_metrics, predict_frames

# Cover unittest discovery before any fixture or native model construction.
source_audit()
install_direct_only_runtime_guard(profile="legacy87")


def identity_fixture():
    width=3
    # Small fixture follows the same6-node dependency order and homogeneous-last convention.
    cores={"core_0":np.eye(width),"core_1":np.zeros((2,width,width)),
        "core_2":np.zeros((2,2,2)),"core_3":np.zeros((width,width,2)),
        "core_4":np.zeros((width,width,width)),"core_5":np.zeros((width,width,width)),"head":np.eye(width)}
    return cores


class NativeTests(unittest.TestCase):
    def test_source_closure_no_reference_factorization(self):
        audit=source_audit()
        self.assertEqual(audit["prohibited_calls_found"],[])
        self.assertFalse(any("odt_ffn_v1" in p or "run_curve" in p for p in audit["source_sha256"]))
    def test_component_metrics_and_invalid_frames(self):
        source={"actions":np.zeros((2,8,7)),"valid":np.array([True,True])}
        output={"actions":np.ones((2,8,7)),"valid":np.array([True,False])}
        metrics,arrays=action_metrics(output,source)
        self.assertEqual(metrics["attempted_invalid_frames"],1)
        self.assertEqual(metrics["translation"]["rmse"],1.)
        self.assertTrue(np.isnan(arrays["action_delta"][1]).all())
    def test_complete_replacement_applies_attention_and_F_once(self):
        import torch
        class Source(torch.nn.Module):
            def __init__(self):
                super().__init__();self.attn_gain=.5
                self.rbn_attn=torch.nn.Identity()
            def attn(self,x,mask=None,method="explicit"):return 2*x
        class F(torch.nn.Module):
            def forward(self,r):return r+3
        wrapper=replacement_block(Source(),F())
        x=torch.ones((1,64,192),dtype=torch.float32)
        torch.testing.assert_close(wrapper(x),torch.full_like(x,5.))
    def test_double_dense_graph_identity_and_denominator_failure(self):
        import torch
        width=193
        # Valid6-node graph: normalized=input, FFNoutput=zero, residualoutput=input.
        cores={"core_0":np.eye(width),"core_1":np.zeros((2,width,width)),
            "core_2":np.zeros((2,2,2)),"core_3":np.zeros((width,width,2)),
            "core_4":np.zeros((width,width,width)),"core_5":np.zeros((width,width,width)),"head":np.eye(width)}
        cores["core_1"][1,-1,-1]=1
        cores["core_2"][:,1,1]=1
        for i in range(width):cores["core_3"][i,i,0]=1
        cores["core_4"][-1,-1,-1]=1
        for i in range(width):cores["core_5"][i,i,-1]=1
        meta={"nodes":[{"children":list(x)} for x in ((),(0,0),(1,1),(0,2),(3,3),(0,4))]}
        module=graph_module(cores,meta,device="cpu",token_batch=2)
        x=torch.arange(384,dtype=torch.float64).reshape(2,192)/400
        torch.testing.assert_close(module(x),x,atol=1e-12,rtol=1e-12)
        bad=cores.copy();bad["head"]=cores["head"].copy();bad["head"][-1]=0
        module=graph_module(bad,meta,device="cpu",token_batch=2)
        with self.assertRaisesRegex(ValueError,"denominator"):
            module(x)
    def test_partial_attempt_counts_do_not_treat_unattempted_as_invalid(self):
        source={"actions":np.zeros((4,8,7)),"valid":np.ones(4,dtype=bool)}
        result={"actions":np.zeros((4,8,7)),"valid":np.array([True,False,False,False]),
            "attempted":np.array([True,True,False,False]),"status_code":np.array([1,3,0,0])}
        metrics,_=action_metrics(result,source)
        self.assertEqual(metrics["attempted_invalid_frames"],1)
        self.assertEqual(metrics["execution_failed_frames"],1)
        self.assertEqual(metrics["not_attempted_frames"],2)
        self.assertIsNone(metrics["invalid_probability_if_fully_attempted"])
        self.assertTrue(metrics["incomplete"])
    def test_actual_stage_hook_and_restoration_after_failure(self):
        import torch
        class Stack(torch.nn.Module):
            def __init__(self):
                super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity()])
        class Vision(torch.nn.Module):
            def __init__(self):super().__init__();self.blocks=Stack()
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.vision=Vision()
            def forward(self,image,tokens,state,embodiment):
                self.vision.blocks.blocks[0](torch.ones((len(image),64,192)))
                return torch.zeros((len(image),8,7)),None
        class Failed(torch.nn.Module):
            def forward(self,x):raise ValueError("nonfinite deliberate execution failure")
        model=Policy();original=model.vision.blocks.blocks[0]
        panel={"image_rgb":np.zeros((2,64,64,3),dtype=np.uint8),
            "image_token_ids":np.zeros((2,32),dtype=np.int64),
            "image_states_normalized":np.zeros((2,8),dtype=np.float32),
            "image_output64":np.full((2,64,192),99,dtype=np.float32),
            "normalization__action_std":np.ones(7,dtype=np.float32),
            "normalization__action_mean":np.zeros(7,dtype=np.float32)}
        result,failure=predict_frames(model,panel,"",device="cpu",batch_size=1)
        self.assertIsNone(failure)
        np.testing.assert_array_equal(result["stage_output"],np.ones((2,64,192)))
        self.assertEqual(len(original._forward_hooks),0)
        failed,failure=predict_frames(model,panel,"",device="cpu",batch_size=1,replacement=Failed())
        self.assertIsNotNone(failure)
        self.assertIs(model.vision.blocks.blocks[0],original)
        self.assertEqual(failed["attempted"].tolist(),[True,False])
        self.assertEqual(failed["status_code"].tolist(),[3,0])
        self.assertEqual(failure["remaining_frames_not_attempted"],1)

    def test_mixed_batch_preserves_true_token_failures_and_valid_neighbor_abort(self):
        import torch
        class Stack(torch.nn.Module):
            def __init__(self):super().__init__();self.blocks=torch.nn.ModuleList([torch.nn.Identity()])
        class Vision(torch.nn.Module):
            def __init__(self):super().__init__();self.blocks=Stack()
        class Policy(torch.nn.Module):
            def __init__(self):super().__init__();self.vision=Vision()
            def forward(self,image,tokens,state,embodiment):
                self.vision.blocks.blocks[0](torch.ones((len(image),64,192)))
                return torch.zeros((len(image),8,7)),None
        class MixedDenominator(torch.nn.Module):
            def forward(self,x):
                self.last_valid=torch.ones(x.shape[:-1],dtype=torch.bool)
                self.last_valid[0,7]=False
                raise ValueError("localFFN projective denominator invalid")
        class Replacement(torch.nn.Module):
            def __init__(self):super().__init__();self.dense=MixedDenominator();self.last_output=None
            def forward(self,x):return self.dense(x)
        panel={"image_rgb":np.zeros((3,64,64,3),dtype=np.uint8),
            "image_token_ids":np.zeros((3,32),dtype=np.int64),
            "image_states_normalized":np.zeros((3,8),dtype=np.float32),
            "normalization__action_std":np.ones(7,dtype=np.float32),
            "normalization__action_mean":np.zeros(7,dtype=np.float32)}
        model=Policy();original=model.vision.blocks.blocks[0]
        output,failure=predict_frames(model,panel,"",device="cpu",batch_size=2,replacement=Replacement())
        self.assertIs(model.vision.blocks.blocks[0],original)
        self.assertEqual(output["attempted"].tolist(),[True,True,False])
        self.assertEqual(output["status_code"].tolist(),[2,4,0])
        self.assertEqual(output["valid"].tolist(),[False,False,False])
        self.assertFalse(output["stage_token_valid"][0,7])
        self.assertEqual(int(np.sum(output["stage_token_valid"][0])),63)
        self.assertTrue(np.all(output["stage_token_valid"][1]))
        self.assertTrue(np.isnan(output["actions_normalized"]).all())
        self.assertEqual(failure["remaining_frames_not_attempted"],1)
        baseline={"actions":np.zeros((3,8,7)),"valid":np.ones(3,dtype=bool)}
        metrics,_=action_metrics(output,baseline)
        self.assertEqual(metrics["denominator_invalid_frames"],1)
        self.assertEqual(metrics["attempted_invalid_frames"],1)
        self.assertEqual(metrics["batch_aborted_frames"],1)
        self.assertEqual(metrics["attempted_unavailable_action_frames"],2)
        self.assertEqual(metrics["not_attempted_frames"],1)
        self.assertIsNone(metrics["invalid_probability_if_fully_attempted"])

    def test_last_batch_abort_is_incomplete_even_when_every_frame_attempted(self):
        source={"actions":np.zeros((2,8,7)),"valid":np.ones(2,dtype=bool)}
        result={"actions":np.full((2,8,7),np.nan),"valid":np.zeros(2,dtype=bool),
            "attempted":np.ones(2,dtype=bool),"status_code":np.array([2,4],dtype=np.int8)}
        metrics,_=action_metrics(result,source)
        self.assertFalse(metrics["validity_measurement_complete"])
        self.assertTrue(metrics["incomplete"])
        self.assertIsNone(metrics["invalid_probability_if_fully_attempted"])
        self.assertEqual(metrics["attempted_invalid_frames"],1)
        self.assertEqual(metrics["batch_aborted_frames"],1)
    def test_runtime_guard(self):
        self.assertEqual(assert_direct_only_runtime_guard(exact_allowed_calls=())["prohibited_attempt_count"],0)


if __name__=="__main__":
    unittest.main()
