"""Prepare17physical local-FFN variants in the accepted NumPy2.2.6 runtime."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np

from research.odt_reference import run_curve as original
from research.odt_reference.run_tests import install_guards, COUNTS, PROHIBITED
from research.odt_reference.curve import RankPlan, rank_schedule, physical_variant
from research.odt_reference.shared_dag import apply_bases
from research.odt_ffn_v1.run import source_audit as producer_audit
from research.odt_ffn_v1.core import ffn_forward
from research.odt_ffn_native_v1.common import (
    CHECKPOINT_SHA, RANKS, SEEDS, DEV_TOKENS, NATIVE_GATE, DOUBLE_GATE,
    digest, require_hash, write_json, read_arrays, conditions, read_panel, dev_selection,
)


def source_audit():
    here = Path(__file__).resolve().parent
    files = ("__init__.py", "common.py", "reference.py", "test_reference.py")
    imports = {"argparse", "ast", "hashlib", "json", "numpy", "unittest", "tempfile"}
    modules = {"pathlib", "research.odt_reference", "research.odt_reference.run_tests",
               "research.odt_reference.curve", "research.odt_reference.shared_dag",
               "research.odt_ffn_v1.run", "research.odt_ffn_v1.core",
               "research.odt_ffn_native_v1.common", "research.odt_ffn_native_v1.reference"}
    for name in files:
        for node in ast.walk(ast.parse((here/name).read_text())):
            if isinstance(node, ast.Import) and any(x.name not in imports for x in node.names):
                raise ValueError("unreviewed FFN reference dependency")
            if isinstance(node, ast.ImportFrom) and node.module not in modules:
                raise ValueError("unreviewed FFN reference import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in PROHIBITED:
                raise ValueError("prohibited FFN bridge call")
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                for a,b in ((node.left,node.right),(node.right,node.left)):
                    if isinstance(b,ast.Attribute) and b.attr == "T" and ast.dump(a) == ast.dump(b.value):
                        raise ValueError("prohibited self-overlap")
    return {"bridge_reference": {name: digest(here/name) for name in files}, "producer": producer_audit()}


def zero_activation(graph, index=4):
    values = []
    for node in graph.nodes[:index+1]:
        if not node.children:
            raw = np.zeros(node.core.shape[1]); raw[-1] = 1.
            value = node.core @ raw
        elif len(node.children) == 1:
            value = node.core @ values[node.children[0]]
        else:
            value = np.einsum("oij,i,j->o", node.core, values[node.children[0]], values[node.children[1]])
        scale = float(np.max(np.abs(value)))
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("zero-input FFN anchor invalid")
        values.append(value/scale)
    return values[-1]


def anchored_basis(graph, bases, seed):
    result = tuple(np.array(b, copy=True) for b in bases)
    anchor = zero_activation(graph)
    columns = np.random.default_rng(seed).normal(size=bases[4].shape)
    columns[:,0] = anchor
    q, triangular = np.linalg.qr(columns, mode="reduced")
    reconstruction = original.close(q@triangular, columns, original.TOLERANCES["coordinates"], "FFN anchor directQR")
    retained = original.close(q[:,0]*triangular[0,0], anchor, original.TOLERANCES["coordinates"], "FFN retained zero anchor")
    if q.shape != bases[4].shape or not np.isfinite(q).all():
        raise ValueError("FFN full directQ completion differs")
    result = result[:4] + (q,) + result[5:]
    return result, {"reconstruction_error": reconstruction, "anchor_error": retained}


def rank_plan(graph, rank):
    base = rank_schedule(graph, percents=(0,)).plans[0]
    if type(rank) is not int or not 1 <= rank <= base.widths[4]:
        raise ValueError("invalid FFN retained width")
    ranks = list(base.widths); ranks[4] = rank
    return RankPlan(-1, tuple(ranks), base.widths, base.eligible, base.signature,
                    base.original_dimensions, base.widths[4]-rank)


def prepare(args):
    sources = source_audit()
    if json.loads(args.source_manifest.read_text()) != sources or np.__version__ != "2.2.6":
        raise ValueError("reference source manifest or NumPy runtime differs")
    install_guards()
    receipt_path = args.producer/"accepted.json"
    require_hash(receipt_path, args.receipt_sha256)
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("accepted") is not True or receipt["checkpoint_sha256"] != CHECKPOINT_SHA
            or receipt["source_sha256"] != sources["producer"]
            or receipt["tolerances"] != original.TOLERANCES or receipt["numpy"] != "2.2.6"
            or receipt["canonical_internal_dimensions"] != 390 or receipt["clone_count"] != 21
            or receipt["environment_binary_exponent"] != 2*receipt["global_binary_exponent"]
            or {r["rank"] for r in receipt["cut_checks"]} != set(RANKS)):
        raise ValueError("accepted localFFN receipt differs")
    graph,bases = original.load_graph(args.producer/"canonical", receipt["canonical_hashes"],
                                      global_exponent=receipt["global_binary_exponent"])
    require_hash(args.producer/"weights.npz", receipt["weights_sha256"])
    weights = read_arrays(args.producer/"weights.npz")
    panel, metadata = read_panel(args.panel, args.panel_metadata)
    for key,value in weights.items():
        if not np.array_equal(value, panel["weight__"+key]):
            raise ValueError("FFNweights differ from native capture checkpoint")
    selected = dev_selection(panel)
    raw = np.asarray([panel["dev_image_post_attention64"][r["image_index"],r["token_index"]] for r in selected],dtype=np.float64)[:,None,:]
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {"schema":"odt-ffn-native-protocol-v1", "conditions":conditions(), "source_sha256":sources,
        "receipt_sha256":args.receipt_sha256, "panel_sha256":digest(args.panel),
        "panel_metadata_sha256":digest(args.panel_metadata), "checkpoint_sha256":CHECKPOINT_SHA,
        "freeze_sha256":metadata["freeze_sha256"], "dev_selection":selected,
        "native_action_gate":NATIVE_GATE,"double_gate":DOUBLE_GATE,"cut_node":4,
        "global_binary_exponent":receipt["global_binary_exponent"],
        "return_to_native_dtype":"float64 decoded F cast to native float32 once",
        "confirmation_frames":160,"development_frames":40,"simulator":False}
    write_json(args.output/"protocol.json",protocol)
    oracle = ffn_forward(raw,weights)
    original.full_rank_replay(graph,original.homogeneous(raw),oracle,"realdev localFFN source")
    np.savez(args.output/"dev_inputs.npz",raw=raw,source_output=oracle)
    zero = original.homogeneous(np.zeros((1,1,192)))
    source_zero = original.checked_chart(graph,zero)
    anchored = {seed:anchored_basis(graph,bases,seed) for seed in SEEDS}
    artifacts = {}
    for condition in conditions():
        full_bases = bases if condition["seed"] is None else anchored[condition["seed"]][0]
        plan = rank_plan(graph,condition["rank"])
        variant = physical_variant(graph,full_bases,plan)
        destination = args.output/condition["name"]
        hashes = original.save_graph(destination,variant.graph,global_exponent=receipt["global_binary_exponent"])
        physical = original.checked_chart(variant.graph,original.homogeneous(raw))
        masked = original.checked_chart(apply_bases(graph,full_bases),original.homogeneous(raw),plan.ranks)
        if physical.valid_rows != masked.valid_rows or physical.failure_reasons != masked.failure_reasons:
            raise ValueError("FFN physical versus mask validity differs")
        replay = original.close(physical.projective,masked.projective,original.TOLERANCES["coordinates"],"FFN everyoccurrence physicalmask")
        anchor_replay = None
        if condition["seed"] is not None:
            cut_zero = original.checked_chart(variant.graph,zero)
            if source_zero.decoded is None or cut_zero.decoded is None:
                raise ValueError("FFN zero-input anchor chart invalid")
            anchor_replay = original.close(cut_zero.decoded,source_zero.decoded,original.TOLERANCES["replay"],"FFN zero wholegraph")
        np.savez(destination/"dev_reference.npz",projective=physical.projective,valid=np.asarray(physical.valid_rows,dtype=bool))
        artifacts[condition["name"]]={"graph_hashes":hashes,"dev_reference_sha256":digest(destination/"dev_reference.npz"),
            "physical_mask_error":replay,"zero_input_replay_error":anchor_replay,
            "removed_dimensions":193-condition["rank"],"eligible_dimensions":390}
        print(json.dumps({"phase":"variant_prepared","condition":condition["name"]}),flush=True)
    require_hash(receipt_path,args.receipt_sha256)
    require_hash(args.panel,protocol["panel_sha256"])
    require_hash(args.panel_metadata,protocol["panel_metadata_sha256"])
    if source_audit()!=sources or COUNTS["prohibited_attempts"]:
        raise ValueError("FFN bridge source drift or prohibited numerical attempt")
    write_json(args.output/"manifest.json",{"ready":True,"schema":"odt-ffn-native-variants-v1",
        "protocol_sha256":digest(args.output/"protocol.json"),"dev_inputs_sha256":digest(args.output/"dev_inputs.npz"),
        "artifacts":artifacts,"kernels":dict(COUNTS)})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--producer",type=Path);parser.add_argument("--receipt-sha256")
    parser.add_argument("--panel",type=Path);parser.add_argument("--panel-metadata",type=Path)
    parser.add_argument("--output",type=Path);parser.add_argument("--source-manifest",type=Path)
    args=parser.parse_args()
    if args.audit_only:
        print(json.dumps(source_audit(),indent=2,sort_keys=True));return
    if any(value is None for name,value in vars(args).items() if name!="audit_only"):
        parser.error("all artifact paths and receipt SHA are required")
    prepare(args)


if __name__=="__main__":
    main()
