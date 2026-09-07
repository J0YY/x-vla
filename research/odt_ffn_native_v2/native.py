"""Float64 localFFN, with separate denominator failures and batch aborts.

The contraction and native policy insertion are unchanged from frozen v1.
Only failed-batch accounting and its schema are revised here.
"""
import argparse
import json
from pathlib import Path
import time
import os
import numpy as np

from scripts.odt_direct_only_compliance import audit_direct_only_launch, assert_direct_only_runtime_guard, install_direct_only_runtime_guard
from scripts.odt_real_panel_v1 import load_native_model, native_environment
from research.odt_ffn_native_v1.common import (
    CHECKPOINT_SHA, DENOMINATOR_MARGIN, NATIVE_GATE, DOUBLE_GATE,
    digest, require_hash, write_json, read_arrays, read_panel, conditions, comparison,
)


def source_audit():
    root=Path(__file__).resolve().parents[2]
    report=audit_direct_only_launch(root,(Path(__file__),Path(__file__).with_name("test_native.py")),require_direct_qr=False)
    for key in ("prohibited_calls_found","prohibited_self_overlap_sites","guarded_dormant_spectral_norm_sites","duplicate_top_level_definition_sites"):
        if report[key]:raise ValueError(f"native FFN source audit failed {key}")
    return report


def load_variant(directory, hashes, exponent, expected_rank):
    require_hash(directory/"graph.json",hashes["graph_sha256"])
    require_hash(directory/"arrays.npz",hashes["arrays_sha256"])
    metadata=json.loads((directory/"graph.json").read_text())
    if metadata["global_binary_exponent"]!=exponent or metadata["arrays_sha256"]!=hashes["arrays_sha256"] or metadata["basis_count"]!=0:
        raise ValueError("physical FFN graph metadata differs")
    arrays=read_arrays(directory/"arrays.npz")
    nodes=metadata["nodes"]
    topology=((),(0,0),(1,1),(0,2),(3,3),(0,4))
    if len(nodes)!=6 or tuple(tuple(n["children"]) for n in nodes)!=topology or set(arrays)!={"head",*(f"core_{i}" for i in range(6))}:
        raise ValueError("physical FFN topology or array inventory differs")
    widths=[]
    for i,node in enumerate(nodes):
        core=arrays[f"core_{i}"]
        if core.dtype!=np.float64 or list(core.shape)!=node["shape"] or not np.isfinite(core).all():
            raise ValueError("physical FFN core dtype/shape/finiteness differs")
        widths.append(core.shape[0])
        expected=(widths[i],)+tuple(widths[j] for j in topology[i]) if i else (193,193)
        if core.shape!=expected:raise ValueError("physical FFN incident dimensions differ")
    if (widths[:4]!=[193,2,2,193] or widths[5]!=193 or widths[4]!=expected_rank or arrays["head"].shape!=(193,193)
            or arrays["head"].dtype!=np.float64 or not np.isfinite(arrays["head"]).all()):
        raise ValueError("only the FFNoutput bond may be narrowed")
    return arrays,metadata


def graph_module(arrays, metadata, *, device, token_batch=16):
    import torch
    class DenseGraph(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cores=tuple(torch.from_numpy(arrays[f"core_{i}"].copy()).to(device) for i in range(6))
            self.head=torch.from_numpy(arrays["head"].copy()).to(device)
            self.children=tuple(tuple(n["children"]) for n in metadata["nodes"])
        def projective(self, raw):
            if raw.dtype!=torch.float64 or raw.ndim!=2 or raw.shape[1]!=192:
                raise ValueError("doubleFFN ingress must be [tokens,192] float64")
            chunks=[]
            for start in range(0,len(raw),token_batch):
                sample=raw[start:start+token_batch]
                ingress=torch.cat((sample,torch.ones((len(sample),1),dtype=torch.float64,device=sample.device)),dim=1)
                values=[]
                for core,children in zip(self.cores,self.children):
                    if not children:value=ingress@core.T
                    elif len(children)==1:value=values[children[0]]@core.T
                    else:
                        left,right=values[children[0]],values[children[1]]
                        intermediate=torch.einsum("oij,bi->boj",core,left)
                        value=(intermediate*right[:,None,:]).sum(dim=-1)
                    if not bool(torch.isfinite(value).all()):raise ValueError("nonfinite localFFN contraction")
                    scale=value.abs().amax(dim=1,keepdim=True)
                    values.append(value/torch.where(scale>0,scale,torch.ones_like(scale)))
                output=values[-1]@self.head.T
                if not bool(torch.isfinite(output).all()):raise ValueError("nonfinite localFFN head")
                scale=output.abs().amax(dim=1,keepdim=True)
                chunks.append(output/torch.where(scale>0,scale,torch.ones_like(scale)))
            return torch.cat(chunks,dim=0)
        def forward(self, raw):
            shape=raw.shape
            projective=self.projective(raw.reshape(-1,192).double())
            valid=projective[:,-1].abs()>DENOMINATOR_MARGIN
            self.last_valid=valid.reshape(shape[:-1]).detach()
            if not bool(valid.all()):raise ValueError("localFFN projective denominator invalid")
            decoded=projective[:,:-1]/projective[:,-1:]
            if not bool(torch.isfinite(decoded).all()):raise ValueError("nonfinite decoded localFFN")
            return decoded.reshape(shape).to(dtype=raw.dtype)
    return DenseGraph()


def replacement_block(original, dense):
    import torch
    class Replacement(torch.nn.Module):
        def __init__(self):
            super().__init__();self.source=original;self.dense=dense
            self.last_output=None
        def forward(self,x,mask=None,method="explicit"):
            r=x+self.source.attn_gain*self.source.attn(self.source.rbn_attn(x),mask=mask,method=method)
            output=self.dense(r)
            self.last_output=output.detach()
            return output
    return Replacement()


def synchronize(device):
    import torch
    if device=="cuda":torch.cuda.synchronize()


def predict_frames(model, panel, prefix, *, device, batch_size, replacement=None):
    import torch
    count=len(panel[prefix+"image_rgb"])
    action=np.full((count,8,7),np.nan,dtype=np.float32)
    stage=np.full((count,64,192),np.nan,dtype=np.float32)
    valid=np.zeros(count,dtype=bool)
    attempted=np.zeros(count,dtype=bool)
    status=np.zeros(count,dtype=np.int8)
    token_valid=np.zeros((count,64),dtype=bool)
    latency=np.full(count,np.nan,dtype=np.float64)
    failure=None
    original=model.vision.blocks.blocks[0]
    stage_capture=[]
    hook=None
    def capture_executed_stage(module,inputs,output):
        stage_capture.append(output.detach())
    if replacement is not None:
        model.vision.blocks.blocks[0]=replacement
    else:
        hook=original.register_forward_hook(capture_executed_stage)
    try:
        with torch.inference_mode():
            for start in range(0,count,batch_size):
                stop=min(start+batch_size,count)
                image=torch.from_numpy(panel[prefix+"image_rgb"][start:stop].copy()).permute(0,3,1,2).to(device).float().div(255)
                tokens=torch.from_numpy(panel[prefix+"image_token_ids"][start:stop]).to(device)
                state=torch.from_numpy(panel[prefix+"image_states_normalized"][start:stop]).to(device)
                synchronize(device);started=time.monotonic()
                attempted[start:stop]=True
                stage_capture.clear()
                # Do not classify this batch using a previous call's mask.
                if replacement is not None and hasattr(replacement,"dense"):
                    replacement.dense.last_valid=None
                    replacement.last_output=None
                try:
                    prediction,loss=model(image,tokens,state,torch.zeros(stop-start,dtype=torch.long,device=device))
                    if loss is not None or not bool(torch.isfinite(prediction).all()):raise ValueError("nonfinite or malformed nativepolicyaction")
                    synchronize(device)
                    action[start:stop]=prediction.cpu().numpy()
                    if replacement is None:
                        if len(stage_capture)!=1 or tuple(stage_capture[0].shape)!=(stop-start,64,192):
                            raise ValueError("native firstblock stage hook count or shape differs")
                        stage[start:stop]=stage_capture[0].cpu().numpy()
                        token_valid[start:stop]=True
                    else:
                        stage[start:stop]=replacement.last_output.cpu().numpy()
                        token_valid[start:stop]=replacement.dense.last_valid.cpu().numpy()
                    valid[start:stop]=True
                    status[start:stop]=1
                    latency[start:stop]=(time.monotonic()-started)/(stop-start)
                except (ValueError,RuntimeError) as error:
                    assert_direct_only_runtime_guard(exact_allowed_calls=())
                    status[start:stop]=3
                    mask=(getattr(replacement.dense,"last_valid",None)
                          if replacement is not None and hasattr(replacement,"dense") else None)
                    if (isinstance(mask,torch.Tensor) and mask.dtype==torch.bool
                            and tuple(mask.shape)==(stop-start,64)):
                        current_tokens=mask.cpu().numpy()
                        token_valid[start:stop]=current_tokens
                        bad_frames=~np.all(current_tokens,axis=1)
                        if "denominator" in str(error) and np.any(bad_frames):
                            # The whole forward call raised, so no action is
                            # available for the otherwise valid neighboring
                            # frame. Preserve that distinction explicitly.
                            status[start:stop]=np.where(bad_frames,2,4)
                    failure={"first_uncompleted_image_index":start,"message":str(error),
                        "attempted_batch_image_indices":list(range(start,stop)),
                        "attempted_batch_status_codes":status[start:stop].tolist(),
                        "remaining_frames_not_attempted":count-stop}
                    break
    finally:
        if hook is not None:hook.remove()
        model.vision.blocks.blocks[0]=original
    denormalized=action*panel["normalization__action_std"]+panel["normalization__action_mean"]
    return {"actions_normalized":action,"actions":denormalized,"stage_output":stage,
            "valid":valid,"attempted":attempted,"status_code":status,"stage_token_valid":token_valid,"latency_s_per_frame":latency},failure


def action_metrics(result, baseline):
    valid=result["valid"]&baseline["valid"]
    delta=result["actions"][valid].astype(np.float64)-baseline["actions"][valid]
    arrays={"action_delta":np.full(result["actions"].shape,np.nan,dtype=np.float64),
            "gripper_sign_disagreement":np.zeros(result["actions"].shape[:2],dtype=bool)}
    arrays["action_delta"][valid]=delta
    arrays["gripper_sign_disagreement"][valid]=(np.sign(result["actions"][valid,:,-1])!=np.sign(baseline["actions"][valid,:,-1]))
    attempted=result.get("attempted",np.ones(len(valid),dtype=bool))
    status=result.get("status_code",np.where(result["valid"],1,2))
    attempted_invalid=attempted & ((status==2)|(status==3))
    complete=bool(np.all(attempted) and not np.any((status==3)|(status==4)))
    metrics={"frames":len(valid),"attempted_invalid_frames":int(np.sum(attempted_invalid)),
        "execution_failed_frames":int(np.sum(attempted & (status==3))),
        "denominator_invalid_frames":int(np.sum(attempted & (status==2))),
        "batch_aborted_frames":int(np.sum(attempted & (status==4))),
        "attempted_unavailable_action_frames":int(np.sum(attempted & ~result["valid"])),
        "not_attempted_frames":int(np.sum(~attempted)),"attempted_frames":int(np.sum(attempted)),
        "validity_measurement_complete":complete,"incomplete":not complete,
        "invalid_probability_if_fully_attempted":float(np.mean(status==2)) if complete else None,
        "jointly_valid_frames":int(np.sum(valid)),"error_statistics_conditional_on_joint_validity":True}
    for label,part in (("translation",slice(0,3)),("rotation",slice(3,6)),("gripper",slice(6,7)),("all",slice(0,7))):
        value=delta[:,:,part]
        per_frame=np.sqrt(np.mean(value*value,axis=(1,2)))
        metrics[label]={"rmse":float(np.sqrt(np.mean(value*value))) if len(value) else None,
            "p95_frame_rmse":float(np.percentile(per_frame,95)) if len(value) else None,
            "maximum_absolute_error":float(np.max(np.abs(value))) if len(value) else None}
    metrics["gripper_sign_disagreement_probability"]=float(np.mean(arrays["gripper_sign_disagreement"][valid])) if np.any(valid) else None
    return metrics,arrays


def execute(args):
    sources=source_audit()
    if json.loads(args.source_manifest.read_text())!=sources:raise ValueError("nativeFFN source manifest differs")
    install_direct_only_runtime_guard(profile="legacy87")
    manifest=json.loads((args.variants/"manifest.json").read_text())
    if manifest.get("ready") is not True or manifest["schema"]!="odt-ffn-native-variants-v1":raise ValueError("variants not accepted")
    if set(manifest["artifacts"])!={condition["name"] for condition in conditions()}:
        raise ValueError("FFN variant artifact inventory differs")
    require_hash(args.variants/"protocol.json",manifest["protocol_sha256"])
    require_hash(args.variants/"dev_inputs.npz",manifest["dev_inputs_sha256"])
    protocol=json.loads((args.variants/"protocol.json").read_text())
    if (protocol["conditions"]!=conditions() or protocol["checkpoint_sha256"]!=CHECKPOINT_SHA
            or protocol["native_action_gate"]!=NATIVE_GATE or protocol["double_gate"]!=DOUBLE_GATE):raise ValueError("FFN native protocol differs")
    require_hash(args.panel,protocol["panel_sha256"]);require_hash(args.panel_metadata,protocol["panel_metadata_sha256"])
    panel,metadata=read_panel(args.panel,args.panel_metadata)
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"protocol.json",{"variants_manifest_sha256":digest(args.variants/"manifest.json"),
        "panel_sha256":digest(args.panel),"source_sha256":sources,"native_action_gate":NATIVE_GATE,
        "double_gate":DOUBLE_GATE,"token_batch":args.token_batch,"image_batch":args.image_batch,
        "conditions":["native"]+[x["name"] for x in conditions()],"score_all160_confirmation_frames":True,
        "failure_accounting_version":2,"supersedes":"v1 failed image batches only",
        "numerical_contraction_and_policy_insertion":"unchanged from v1"})
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"]!=":4096:8":raise ValueError("deterministicCUDA config differs")
    environment=native_environment(args.device)
    model,model_record=load_native_model(args.checkpoint,args.device)
    import torch
    dev_reference=read_arrays(args.variants/"dev_inputs.npz")
    dev_native,dev_failure=predict_frames(model,panel,"dev_",device=args.device,batch_size=args.image_batch)
    replay=comparison(dev_native["actions_normalized"],panel["dev_image_actions_normalized"])
    if dev_failure or not replay["passed"]:raise ValueError("reloaded nativepolicy dev parity failed")
    outcomes={};baseline=None;fullrank_source_gate=None
    for condition in [{"name":"native"}]+conditions():
        name=condition["name"]
        dense=wrapper=None;double_replay=None
        if name!="native":
            artifact=manifest["artifacts"][name]
            directory=args.variants/name
            arrays,description=load_variant(directory,artifact["graph_hashes"],protocol["global_binary_exponent"],condition["rank"])
            dense=graph_module(arrays,description,device=args.device,token_batch=args.token_batch)
            require_hash(directory/"dev_reference.npz",artifact["dev_reference_sha256"])
            reference=read_arrays(directory/"dev_reference.npz")
            with torch.inference_mode():
                observed=dense.projective(torch.from_numpy(dev_reference["raw"][:,0]).to(args.device)).cpu().numpy()
            double_replay=comparison(observed,reference["projective"],double=True)
            observed_valid=np.abs(observed[:,-1])>DENOMINATOR_MARGIN
            if not np.array_equal(observed_valid,reference["valid"]) or not double_replay["passed"]:
                outcomes[name]={"completed":False,"phase":"double_dev_gate","parity":double_replay}
                if name=="fullrank":
                    write_json(args.output/"fullrank_double_failure.json",outcomes[name])
                    raise ValueError("fullrank double projective gate failed")
                write_json(args.output/(name+".json"),dict(outcomes[name],confirmation_not_attempted=True))
                continue
            if name=="fullrank":
                decoded=observed[:,:-1]/observed[:,-1:]
                source_gate=comparison(decoded,dev_reference["source_output"],double=True)
                fullrank_source_gate=source_gate
                if not source_gate["passed"]:raise ValueError("fullrank Torch F versus independent source gate failed")
            wrapper=replacement_block(model.vision.blocks.blocks[0],dense)
        development,failure=(dev_native,dev_failure) if name=="native" else predict_frames(model,panel,"dev_",device=args.device,batch_size=args.image_batch,replacement=wrapper)
        dev_action_gate=comparison(development["actions_normalized"],dev_native["actions_normalized"]) if name in ("native","fullrank") else None
        dev_denormalized_gate=comparison(development["actions"],dev_native["actions"]) if name in ("native","fullrank") else None
        if failure or (dev_action_gate is not None and (not dev_action_gate["passed"] or not dev_denormalized_gate["passed"])):
            outcomes[name]={"completed":False,"phase":"development","failure":failure,"native_action_parity":dev_action_gate}
            np.savez(args.output/(name+"_development.npz"),**development)
            outcomes[name]["development_sha256"]=digest(args.output/(name+"_development.npz"))
            outcomes[name]["confirmation_not_attempted"]=True
            write_json(args.output/(name+".json"),outcomes[name])
            if name=="fullrank":
                write_json(args.output/"fullrank_gate_failure.json",outcomes[name]);raise ValueError("fullrank nativeaction gate failed")
            continue
        result,failure=predict_frames(model,panel,"",device=args.device,batch_size=args.image_batch,replacement=wrapper)
        if name=="native":
            baseline=result
            gate=comparison(result["actions_normalized"],panel["image_actions_normalized"])
            if failure or not gate["passed"]:raise ValueError("confirmation native reload parity failed")
        if name=="fullrank":
            gate=comparison(result["actions_normalized"],baseline["actions_normalized"])
            denormalized_gate=comparison(result["actions"],baseline["actions"])
            if failure or not gate["passed"] or not denormalized_gate["passed"]:
                np.savez(args.output/(name+".npz"),**result)
                write_json(args.output/"fullrank_confirmation_failure.json",{"parity":gate,"failure":failure})
                raise ValueError("fullrank nativeaction confirmationgate failed")
            fullrank=result
        reference_baseline=baseline if name in ("native","fullrank") else fullrank
        metrics,extra=action_metrics(result,reference_baseline)
        output_arrays={**result,**extra,"task_ids":panel["image_task_ids"],"episode_ids":panel["image_episode_ids"],"frame_ids":panel["image_frame_ids"]}
        np.savez(args.output/(name+".npz"),**output_arrays)
        np.savez(args.output/(name+"_development.npz"),**development)
        outcomes[name]={"completed":failure is None,"failure":failure,"metrics":metrics,
            "compared_against":"native" if name in ("native","fullrank") else "fullrank",
            "double_dev_parity":double_replay,"native_action_dev_parity":dev_action_gate,
            "native_denormalized_action_dev_parity":dev_denormalized_gate,
            "fullrank_source_double_parity":fullrank_source_gate if name=="fullrank" else None,
            "native_denormalized_confirmation_parity":denormalized_gate if name=="fullrank" else None,
            "native_action_confirmation_parity":gate if name in ("native","fullrank") else None,
            "array_sha256":digest(args.output/(name+".npz")),"development_sha256":digest(args.output/(name+"_development.npz")),
            "mean_latency_s_per_frame":float(np.nanmean(result["latency_s_per_frame"])) if np.any(result["valid"]) else None}
        write_json(args.output/(name+".json"),outcomes[name])
        print(json.dumps({"phase":"arm_complete","name":name,"completed":failure is None,"metrics":metrics}),flush=True)
    runtime=assert_direct_only_runtime_guard(exact_allowed_calls=())
    if source_audit()!=sources:raise ValueError("nativeFFN source drift")
    write_json(args.output/"manifest.json",{"workflow_completed":True,
        "all_arms_completed":all(outcome["completed"] for outcome in outcomes.values()) and len(outcomes)==18,
        "failed_or_stopped_arms":[name for name,outcome in outcomes.items() if not outcome["completed"]],
        "not_attempted_arms":[condition["name"] for condition in [{"name":"native"}]+conditions() if condition["name"] not in outcomes],
        "schema":"odt-ffn-native-actions-v2","outcomes":outcomes,
        "environment":environment,"model":model_record,"runtime_guard":runtime,"source_sha256":sources,
        "variants_manifest_sha256":digest(args.variants/"manifest.json"),"panel_sha256":digest(args.panel),
        "status_codes":{"not_attempted":0,"valid":1,"denominator_invalid":2,"execution_failure":3,"batch_aborted":4},
        "failure_accounting_version":2,"supersedes":"v1 failed image batches only",
        "simulator_launched":False,"full_policy_odt":False,"intervention_scope":"one local residualFFN at all64 first-block tokens"})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only",action="store_true")
    for name in ("variants","panel","panel-metadata","checkpoint","source-manifest","output"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--token-batch",type=int,default=16);parser.add_argument("--image-batch",type=int,default=2)
    args=parser.parse_args()
    if args.audit_only:print(json.dumps(source_audit(),indent=2,sort_keys=True));return
    if any(value is None for value in vars(args).values()) or not 1<=args.token_batch<=32 or not 1<=args.image_batch<=8:
        parser.error("allpathsrequired,tokenbatch1..32,imagebatch1..8")
    execute(args)


if __name__=="__main__":main()
