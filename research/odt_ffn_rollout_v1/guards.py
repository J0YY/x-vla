"""Compose the frozen controller guard before the final native guard owner."""
import ast
import inspect
from pathlib import Path
import numpy as np

from scripts.odt_direct_only_compliance import install_direct_only_runtime_guard, assert_direct_only_runtime_guard
from research.odt_ffn_native_v1.native import source_audit as native_audit
from research.odt_ffn_native_v2.native import source_audit as offline_audit
from research.odt_ffn_native_v1.common import digest, require_hash
from research.odt_ffn_rollout_v1 import controller

CONTROLLER_SHA="e89c2b9efe953086859a5c9f37fcbca0fef1fcf9dd9f8c301fe94ce61aaf2687"
FIXTURES={"modal_odt_controller_real_gate_capture.json":"d90397a8f1aa4bdf86cd07a8f163df12956519a359ff2b9cfdd0c410a6c4b3b7",
    "modal_odt_controller_zero_component_capture.json":"1d00f4e60c8713d0dcf5edefa18c330c36e66cf59cf49faa2483adc30c85b5a1"}
PROHIBITED={"svd","svdvals","svd_lowrank","pca_lowrank","pinv","pinvh","pinverse","lstsq","polar","cov","covariance","gram","matrix_rank","cond","orth","null_space","inv","inverse","solve","cholesky","matrix_power","det","slogdet"}
STATE={"installed":False,"qr_calls":0,"prohibited_qr_callers":0}
SCIPY_WRAPPERS={}
EXTRA_WRAPPERS={}
QR_WRAPPER=None
CONTROLLER_GUARD=None


def source_audit():
    here=Path(__file__).resolve().parent
    require_hash(here/"controller.py",CONTROLLER_SHA)
    for name,expected in FIXTURES.items():require_hash(here/"fixtures"/name,expected)
    files=("__init__.py","guards.py","worker.py","test_controller.py","test_worker.py")
    imports={"argparse","ast","inspect","json","math","numbers","hashlib","os","platform","random","time","sys","unittest","tempfile","numpy","torch","scipy.linalg","importlib.metadata","pytest","robosuite.controllers.osc","robosuite.utils.control_utils"}
    modules={"__future__","pathlib","unittest.mock","PIL","scripts.odt_direct_only_compliance",
        "scripts.odt_real_panel_v1","research.odt_ffn_native_v1.native","research.odt_ffn_native_v2.native","research.odt_ffn_native_v1.common",
        "research.odt_ffn_rollout_v1","research.odt_ffn_rollout_v1.controller","research.odt_ffn_rollout_v1.guards","research.odt_ffn_rollout_v1.worker",
        "libero.libero","libero.libero.envs","robosuite.utils.transform_utils"}
    for name in files:
        for node in ast.walk(ast.parse((here/name).read_text())):
            if isinstance(node,ast.Import) and any(x.name not in imports for x in node.names):raise ValueError("unreviewed rollout dependency")
            if isinstance(node,ast.ImportFrom) and node.module not in modules:raise ValueError("unreviewed rollout import")
            if isinstance(node,ast.Attribute) and node.attr in PROHIBITED:raise ValueError("prohibited rollout binding")
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in PROHIBITED|{"qr","eigh","eig","eigvals","eigvalsh","norm","matrix_norm"}:
                raise ValueError("prohibited rollout numerical call")
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.MatMult):
                for a,b in ((node.left,node.right),(node.right,node.left)):
                    if isinstance(b,ast.Attribute) and b.attr=="T" and ast.dump(a)==ast.dump(b.value):raise ValueError("prohibited self-overlap")
    return {"local":{name:digest(here/name) for name in (*files,"tasks.json","controller.py")},
        "fixtures":FIXTURES,"native_source_audit":native_audit(),"offline_source_audit":offline_audit(),
        "controller_admission":"exact reviewed byte identity, unchanged NumPy directQR caller"}


def install():
    global CONTROLLER_GUARD,QR_WRAPPER
    if STATE["installed"]:return assert_intact()
    require_hash(Path(controller.__file__),CONTROLLER_SHA)
    # This must precede legacy87. The latter must remain the final shared owner.
    CONTROLLER_GUARD=controller.install_prohibited_route_guards()
    import scipy.linalg
    for name in ("svd","svdvals","pinv","pinvh","lstsq","polar","orth","null_space"):
        if hasattr(scipy.linalg,name):
            wrapper=getattr(scipy.linalg,name)
            if wrapper.__name__!="reject":raise ValueError("SciPy controller guard not installed")
            SCIPY_WRAPPERS[name]=wrapper
    SCIPY_WRAPPERS["norm"]=scipy.linalg.norm
    if scipy.linalg.norm.__name__!="wrapper":raise ValueError("SciPy spectralnorm guard not installed")
    native=install_direct_only_runtime_guard(profile="legacy87")
    import torch
    namespaces={"numpy":np,"numpy.linalg":np.linalg,"numpy.linalg.linalg":np.linalg.linalg,
        "scipy.linalg":scipy.linalg,"torch":torch,"torch.linalg":torch.linalg,"torch.Tensor":torch.Tensor}
    for recorded in CONTROLLER_GUARD["blocked_entrypoints"]:
        qualified=recorded.replace(".norm_spectral_matrix_orders",".norm")
        if qualified in native["patched_entrypoints"]:continue
        namespace,name=qualified.rsplit(".",1)
        owner=namespaces[namespace]
        wrapper=getattr(owner,name)
        if wrapper.__name__ not in {"reject","wrapper"}:raise ValueError("extra controller guard owner differs")
        EXTRA_WRAPPERS[qualified]=(owner,name,wrapper)
    original=np.linalg.qr
    def direct_controller_qr(*args,**kwargs):
        caller=inspect.currentframe().f_back
        if caller.f_code is not controller._direct_qr_square_solve.__code__:
            STATE["prohibited_qr_callers"]+=1
            raise RuntimeError("NumPy QR called outside the frozen controller solver")
        STATE["qr_calls"]+=1
        return original(*args,**kwargs)
    QR_WRAPPER=direct_controller_qr
    np.linalg.qr=direct_controller_qr
    STATE["installed"]=True
    return assert_intact()


def assert_intact():
    import scipy.linalg
    native=assert_direct_only_runtime_guard(exact_allowed_calls=())
    if not STATE["installed"] or np.linalg.qr is not QR_WRAPPER or STATE["prohibited_qr_callers"]:
        raise RuntimeError("rollout QR guard ownership differs")
    if any(getattr(scipy.linalg,name) is not wrapper for name,wrapper in SCIPY_WRAPPERS.items()):
        raise RuntimeError("rollout SciPy guard ownership differs")
    if any(getattr(owner,name) is not wrapper for owner,name,wrapper in EXTRA_WRAPPERS.values()):
        raise RuntimeError("controller-only guard ownership differs")
    if CONTROLLER_GUARD["prohibited_calls"] or CONTROLLER_GUARD["matrix_spectral_norm_calls"]:
        raise RuntimeError("controller prohibited numerical attempt")
    return {"native":native,"controller":dict(CONTROLLER_GUARD),"controller_qr":dict(STATE),
        "controller_only_verified_entrypoints":sorted(EXTRA_WRAPPERS),
        "composition":"controllerfirst,legacy87last,privatecontroller-onlyNumPyQRwrapper"}


def patch_controller():
    receipt=controller.install_controller()
    assert_controller_binding()
    return receipt


def assert_controller_binding():
    import robosuite.controllers.osc as osc
    import robosuite.utils.control_utils as utilities
    if osc.opspace_matrices is not controller.opspace_matrices_direct_qr or utilities.opspace_matrices is not controller.opspace_matrices_direct_qr:
        raise RuntimeError("stock controller binding returned")
    assert_intact()
