"""Smoke allfourarms, then run one task/two identical initializations perworker."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import numbers
import os
from pathlib import Path
import random
import time
import numpy as np

from scripts.odt_real_panel_v1 import load_native_model,native_environment,load_training,encode_instruction
from research.odt_ffn_native_v1.native import load_variant,graph_module,replacement_block
from research.odt_ffn_native_v2.native import source_audit as offline_source_audit
from research.odt_ffn_native_v1.common import CHECKPOINT_SHA,digest,require_hash,read_arrays,write_json
from research.odt_ffn_rollout_v1 import controller
from research.odt_ffn_rollout_v1.guards import source_audit,install,assert_intact,patch_controller,assert_controller_binding

ARMS=("native","fullrank","leading_k174","anchored_k174_s0")
MAX_STEPS=280
SETTLE_STEPS=10
HORIZON=8
SMOKE_MAX_PROJECTED_SECONDS=3000.


def array_digest(value):
    value=np.ascontiguousarray(np.asarray(value))
    result=hashlib.sha256(str(value.dtype).encode()+b"\0"+json.dumps(list(value.shape)).encode()+b"\0")
    result.update(value.tobytes());return result.hexdigest()


def task_authority():
    return json.loads((Path(__file__).parent/"tasks.json").read_text())


def offline_gate(args):
    require_hash(args.offline/"manifest.json",args.offline_sha256)
    offline=json.loads((args.offline/"manifest.json").read_text())
    if offline.get("workflow_completed") is not True or offline.get("schema")!="odt-ffn-native-actions-v2" or offline.get("failure_accounting_version")!=2:
        raise ValueError("offline nativeaction workflow did not complete")
    if offline.get("source_sha256")!=offline_source_audit():raise ValueError("offline v2 source closure differs")
    publication=json.loads((args.offline/"protocol.json").read_text())
    if (publication.get("failure_accounting_version")!=2
            or any(publication.get(key)!=offline.get(key) for key in ("source_sha256","variants_manifest_sha256","panel_sha256"))
            or publication.get("score_all160_confirmation_frames") is not True):
        raise ValueError("offline v2 publication protocol differs")
    if any(arm in offline.get("failed_or_stopped_arms",[]) or arm in offline.get("not_attempted_arms",[]) for arm in ARMS):
        raise ValueError("required offline arm was failed or not attempted")
    if offline["model"]["checkpoint_sha256"]!=CHECKPOINT_SHA:raise ValueError("offline checkpoint identity differs")
    for arm in ARMS:
        record=offline["outcomes"].get(arm,{})
        if record.get("completed") is not True or record.get("failure") is not None:raise ValueError("required offline arm did not pass")
        if json.loads((args.offline/(arm+".json")).read_text())!=record:
            raise ValueError("offline arm publication differs from manifest")
        for suffix,count,key in (("",160,"array_sha256"),("_development",40,"development_sha256")):
            path=args.offline/(arm+suffix+".npz")
            require_hash(path,record[key]);arrays=read_arrays(path)
            if (arrays["valid"].shape!=(count,) or arrays["attempted"].shape!=(count,)
                    or arrays["status_code"].shape!=(count,) or not np.all(arrays["valid"])
                    or not np.all(arrays["attempted"]) or not np.all(arrays["status_code"]==1)
                    or arrays["stage_token_valid"].shape!=(count,64) or not np.all(arrays["stage_token_valid"])):
                raise ValueError("required offline arm has invalid or unattempted frames")
            for action_key in ("actions","actions_normalized"):
                if arrays[action_key].shape!=(count,8,7) or not np.isfinite(arrays[action_key]).all():
                    raise ValueError("required offline arm action array differs")
    fullrank=offline["outcomes"]["fullrank"]
    for key in ("double_dev_parity","native_action_dev_parity","native_action_confirmation_parity",
                "native_denormalized_action_dev_parity","native_denormalized_confirmation_parity","fullrank_source_double_parity"):
        if fullrank.get(key,{}).get("passed") is not True:raise ValueError("fullrank nativepolicyaction parity absent")
    require_hash(args.variants/"manifest.json",offline["variants_manifest_sha256"])
    manifest=json.loads((args.variants/"manifest.json").read_text())
    require_hash(args.variants/"protocol.json",manifest["protocol_sha256"])
    protocol=json.loads((args.variants/"protocol.json").read_text())
    if protocol["checkpoint_sha256"]!=CHECKPOINT_SHA:raise ValueError("physicalvariant checkpoint differs")
    return manifest,protocol


def numerical_environment():
    expected=task_authority()["EXPECTED_VERSIONS"]
    actual={name:importlib.metadata.version(name) for name in ("libero","robosuite","mujoco","pillow")}
    if any(actual[name]!=expected[name] for name in actual):raise ValueError(f"simulator version identity differs: {actual}")
    if os.environ.get("MUJOCO_GL")!="egl":raise ValueError("headlessEGL required")
    return actual


def packaged_states(suite,index):
    import torch
    from libero.libero import get_libero_path
    original_load=torch.load
    root=Path(get_libero_path("init_states")).resolve()
    opened=[]
    def trusted_load(path,*args,**kwargs):
        value=Path(path)
        if value.is_symlink() or not value.is_file() or root not in value.resolve().parents:
            raise ValueError("initialstates path outside trusted packaged root")
        opened.append({"path":str(value.resolve()),"sha256":digest(value)})
        kwargs["weights_only"]=False
        return original_load(path,*args,**kwargs)
    torch.load=trusted_load
    try:states=suite.get_task_init_states(index)
    finally:torch.load=original_load
    states=np.asarray(states)
    expected=task_authority()["EXPECTED_TASK_PROTOCOL"][str(index)]
    if len(opened)!=1 or len(states)!=50 or array_digest(states)!=expected[3]:raise ValueError("official initialstate bundle differs")
    if len({array_digest(row) for row in states})!=50:raise ValueError("packaged initialstates are duplicated")
    return states,opened[0]


def prepare_task(index):
    from libero.libero import benchmark,get_libero_path
    suite=benchmark.get_benchmark_dict()["libero_object"]()
    if suite.n_tasks!=10:raise ValueError("LIBERO task count differs")
    expected=task_authority()["EXPECTED_TASK_PROTOCOL"]
    if any(str(suite.get_task(i).language)!=expected[str(i)][0] for i in range(10)):
        raise ValueError("LIBERO task ordering differs")
    task=suite.get_task(index)
    bddl=Path(get_libero_path("bddl_files"))/task.problem_folder/task.bddl_file
    if task.problem_folder!="libero_object" or task.bddl_file!=expected[str(index)][1]:raise ValueError("BDDL task identity differs")
    require_hash(bddl,expected[str(index)][2])
    states,package=packaged_states(suite,index)
    return str(task.language),bddl,states,{"task_index":index,"language":str(task.language),
        "bddl_sha256":digest(bddl),"init_states_array_sha256":array_digest(states),"packaged_file":package}



def validate_observation(observation,label="policy"):
    if not isinstance(observation,dict):raise RuntimeError(f"{label} observation is not a mapping")
    expected={"agentview_image":((64,64,3),np.uint8),"robot0_eef_pos":((3,),np.float64),
        "robot0_eef_quat":((4,),np.float64),"robot0_gripper_qpos":((2,),np.float64)}
    for field,(shape,dtype) in expected.items():
        if field not in observation:raise RuntimeError(f"{label} observation omits {field}")
        try:value=np.asarray(observation[field])
        except (TypeError,ValueError) as error:raise RuntimeError(f"{label} observation {field} is malformed") from error
        if value.shape!=shape or value.dtype!=dtype or not np.isfinite(value).all():
            raise RuntimeError(f"{label} observation {field} shape, dtype, or finiteness differs")
    return observation


def validated_transition(value,label):
    if not isinstance(value,tuple) or len(value)!=4:raise RuntimeError(f"{label} transition schema differs")
    observation,reward,done,_info=value
    if type(done) not in (bool,np.bool_):raise RuntimeError(f"{label} done flag is not a literal boolean")
    if isinstance(reward,(bool,np.bool_)) or not isinstance(reward,numbers.Real) or not math.isfinite(float(reward)):
        raise RuntimeError(f"{label} reward is not a finite scalar")
    return validate_observation(observation,label),float(reward),bool(done)


def state_vector(observation):
    validate_observation(observation,"robot-state input")
    from robosuite.utils.transform_utils import quat2axisangle
    value=np.concatenate((np.asarray(observation["robot0_eef_pos"],dtype=np.float64),
        quat2axisangle(observation["robot0_eef_quat"]),np.asarray(observation["robot0_gripper_qpos"],dtype=np.float64))).astype(np.float32)
    if value.shape!=(8,) or not np.isfinite(value).all():raise ValueError("malformed robot state")
    return value


def rgb_image(observation):
    validate_observation(observation,"image input")
    from PIL import Image
    rotated=np.ascontiguousarray(observation["agentview_image"][::-1,::-1])
    return np.asarray(Image.fromarray(rotated).resize((64,64))).copy()


def predict(model,observation,instruction,normalization,device):
    import torch
    image=rgb_image(observation);raw_state=state_vector(observation)
    tensor=torch.from_numpy(image).permute(2,0,1).to(device).float().div(255).unsqueeze(0)
    state=torch.from_numpy((raw_state-normalization["state_mean"])/normalization["state_std"]).to(device).unsqueeze(0)
    with torch.inference_mode():
        output,loss=model(tensor,instruction,state,torch.zeros(1,dtype=torch.long,device=device))
    if loss is not None or tuple(output.shape)!=(1,8,7) or not bool(torch.isfinite(output).all()):raise ValueError("nonfinite or malformed nativepolicyaction")
    normalized=output[0].cpu().numpy()
    return normalized,normalized*normalization["action_std"]+normalization["action_mean"]


def episode(simulation,model,initial_state,instruction,normalization,*,device,seed):
    assert_controller_binding()
    simulation.seed(seed)
    validate_observation(simulation.reset(),"reset")
    observation=validate_observation(simulation.set_init_state(initial_state),"initial-state")
    for settle_index in range(SETTLE_STEPS):
        observation,reward,done=validated_transition(simulation.step([0.,0.,0.,0.,0.,0.,-1.]),f"settle step {settle_index}")
        if done or reward>0:raise RuntimeError("terminaltransition during fixedsettling")
    start=time.monotonic();steps=0;success=False;terminated=False
    images=[];states=[];actions=[];normalized_actions=[];rewards=[];terminals=[]
    prediction_steps=[];predictions=[];denormalized_predictions=[]
    failure=None
    while steps<MAX_STEPS and not success and not terminated:
        validate_observation(observation,f"policy input {steps}")
        try:normalized,chunk=predict(model,observation,instruction,normalization,device)
        except (ValueError,RuntimeError) as error:
            assert_intact()
            failure={"step":steps,"message":str(error),"kind":"policy_denominator" if "denominator invalid" in str(error) else "experiment_failure"}
            break
        prediction_steps.append(steps);predictions.append(normalized);denormalized_predictions.append(chunk)
        for offset in range(HORIZON):
            action=chunk[offset].copy();action[-1]=1. if action[-1]>0 else -1.
            images.append(rgb_image(observation));states.append(state_vector(observation))
            actions.append(action.copy());normalized_actions.append(normalized[offset].copy())
            observation,reward,done=validated_transition(simulation.step(action.tolist()),f"policy step {steps}")
            steps+=1;success=bool(reward>0);terminated=bool(done)
            rewards.append(float(reward));terminals.append(terminated)
            if success or terminated or steps>=MAX_STEPS:break
    assert_controller_binding()
    elapsed=time.monotonic()-start
    arrays={"rgb":np.asarray(images,dtype=np.uint8).reshape(-1,64,64,3),"states_raw":np.asarray(states,dtype=np.float32).reshape(-1,8),
        "actions_executed":np.asarray(actions,dtype=np.float32).reshape(-1,7),"actions_normalized":np.asarray(normalized_actions,dtype=np.float32).reshape(-1,7),
        "rewards":np.asarray(rewards,dtype=np.float64),"terminals":np.asarray(terminals,dtype=bool),
        "prediction_steps":np.asarray(prediction_steps,dtype=np.int64),"action_chunks_normalized":np.asarray(predictions,dtype=np.float32).reshape(-1,8,7),
        "action_chunks_denormalized":np.asarray(denormalized_predictions,dtype=np.float32).reshape(-1,8,7),
        "executed_step_action_valid":np.ones(steps,dtype=bool),"initial_state":np.asarray(initial_state)}
    return arrays,{"success":success,"steps":steps,"terminated_without_success":terminated and not success,
        "measurement_complete":failure is None,"failure":failure,"elapsed_s":elapsed,"seed":seed,
        "initial_state_sha256":array_digest(initial_state),"controller_counts":dict(controller.COUNTS)}


def run_arm(args,model,arm,index,episodes,manifest,protocol,vocab,normalization):
    import torch
    from libero.libero.envs import OffScreenRenderEnv
    setup=time.monotonic()
    language,bddl,states,task_record=prepare_task(index)
    instruction=torch.tensor([encode_instruction(language,vocab)],dtype=torch.long,device=args.device)
    original=model.vision.blocks.blocks[0]
    if arm!="native":
        condition=next(x for x in protocol["conditions"] if x["name"]==arm)
        arrays,metadata=load_variant(args.variants/arm,manifest["artifacts"][arm]["graph_hashes"],protocol["global_binary_exponent"],condition["rank"])
        dense=graph_module(arrays,metadata,device=args.device,token_batch=16)
        model.vision.blocks.blocks[0]=replacement_block(original,dense)
    simulation=None;records=[]
    try:
        assert_controller_binding()
        simulation=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=64,camera_widths=64)
        assert_controller_binding()
        setup_elapsed=time.monotonic()-setup
        for index_episode in episodes:
            try:
                arrays,record=episode(simulation,model,states[index_episode],instruction,normalization,
                                      device=args.device,seed=index*100+index_episode)
            except (ValueError,RuntimeError) as error:
                write_json(args.output/f"{arm}_task{index}_episode{index_episode}_experiment_failure.json",
                    {"measurement_complete":False,"ordinary_unsuccessful_episode":False,"arm":arm,"task_index":index,
                     "episode":index_episode,"message":str(error),"controller_counts":dict(controller.COUNTS)})
                raise
            name=f"{arm}_task{index}_episode{index_episode}"
            np.savez_compressed(args.output/(name+".npz"),**arrays)
            record.update({"arm":arm,"task_index":index,"episode":index_episode,"arrays_sha256":digest(args.output/(name+".npz")),
                "task_protocol":task_record,"setup_elapsed_s":setup_elapsed})
            write_json(args.output/(name+".json"),record);records.append(record)
            print(json.dumps(record,sort_keys=True),flush=True)
            if not record["measurement_complete"]:break
    finally:
        model.vision.blocks.blocks[0]=original
        if simulation is not None:simulation.close()
    return records


def projection(records):
    if len(records)!=4 or {r["arm"] for r in records}!=set(ARMS) or any(not r["measurement_complete"] for r in records):
        return {"passed":False,"reason":"incomplete smoke arms"}
    seconds=1.25*20*sum(r["setup_elapsed_s"]+MAX_STEPS*r["elapsed_s"]/max(1,r["steps"]) for r in records)
    return {"passed":seconds<=SMOKE_MAX_PROJECTED_SECONDS,"projected_serial80_episode_seconds":seconds,
        "maximum_seconds":SMOKE_MAX_PROJECTED_SECONDS,"formula":"1.25*20*sum_arm(setup+280*episode_elapsed/max(1,steps))"}


def execute(args):
    sources=source_audit()
    if json.loads(args.source_manifest.read_text())!=sources:raise ValueError("rollout source manifest differs")
    install()
    manifest,protocol=offline_gate(args)
    if args.mode=="worker":
        require_hash(args.smoke,args.smoke_sha256)
        smoke=json.loads(args.smoke.read_text())
        if (smoke.get("schema")!="odt-ffn-rollout-smoke-v1" or smoke.get("timing",{}).get("passed") is not True
                or smoke["source_sha256"]!=sources or smoke["offline_sha256"]!=args.offline_sha256
                or smoke["variants_manifest_sha256"]!=digest(args.variants/"manifest.json")):
            raise ValueError("rollout smoke/timing admission failed")
    args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/"protocol.json",{"mode":args.mode,"source_sha256":sources,"offline_sha256":args.offline_sha256,
        "arms":ARMS,"max_steps":MAX_STEPS,"settle_steps":SETTLE_STEPS,"action_horizon":HORIZON,
        "tasks":10,"episodes_per_task":2,"image":"rot180+PILdefaultresize64+float32/255",
        "gripper":"positive=>1,otherwise-1","all_arms_same_controller":True,"full_policy_odt":False})
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    os.environ.setdefault("MUJOCO_GL","egl")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"]!=":4096:8":raise ValueError("CUDA workspace config differs")
    environment=native_environment(args.device)
    environment.update(numerical_environment())
    random.seed(0);np.random.seed(0)
    model,model_record=load_native_model(args.checkpoint,args.device)
    vocab,normalization=load_training(args.training)
    # Bind before any simulator object exists, and verify after construction.
    controller_record=patch_controller()
    records=[]
    if args.mode=="smoke":
        for arm in ARMS:
            records.extend(run_arm(args,model,arm,0,(0,),manifest,protocol,vocab,normalization))
            if not records[-1]["measurement_complete"]:break
    else:
        records=run_arm(args,model,args.arm,args.task_index,(0,1),manifest,protocol,vocab,normalization)
    guards=assert_intact()
    if source_audit()!=sources:raise ValueError("rollout source changed")
    receipt={"schema":"odt-ffn-rollout-smoke-v1" if args.mode=="smoke" else "odt-ffn-rollout-worker-v1",
        "measurement_complete":len(records)==(4 if args.mode=="smoke" else 2) and all(r["measurement_complete"] for r in records),
        "records":records,"source_sha256":sources,"environment":environment,"model":model_record,"controller":controller_record,
        "guards":guards,"offline_sha256":args.offline_sha256,"variants_manifest_sha256":digest(args.variants/"manifest.json"),
        "full_policy_odt":False,"pilot_only":True}
    if args.mode=="smoke":receipt["timing"]=projection(records)
    write_json(args.output/"receipt.json",receipt)
    if not receipt["measurement_complete"]:raise RuntimeError("rollout measurement stopped, inspect failure receipt")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--mode",choices=("smoke","worker"))
    for name in ("source-manifest","offline","variants","checkpoint","training","output","smoke"):
        parser.add_argument("--"+name,type=Path)
    parser.add_argument("--offline-sha256");parser.add_argument("--smoke-sha256")
    parser.add_argument("--arm",choices=ARMS);parser.add_argument("--task-index",type=int)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args=parser.parse_args()
    if args.audit_only:print(json.dumps(source_audit(),indent=2,sort_keys=True));return
    required=(args.mode,args.source_manifest,args.offline,args.offline_sha256,args.variants,args.checkpoint,args.training,args.output)
    if any(value is None for value in required):parser.error("all artifact paths and offline SHA required")
    if args.mode=="worker" and (args.arm is None or args.task_index is None or not 0<=args.task_index<10 or args.smoke is None or args.smoke_sha256 is None):
        parser.error("worker requires one task0..9, arm, authenticated smoke")
    execute(args)


if __name__=="__main__":main()
