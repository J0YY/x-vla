"""Small dense reference for Dooms et al., arXiv:2504.02667, Algorithms 1–3.

Convention: embedding[hidden,input], core[out,left,right], head[output,hidden].
The object is a locally symmetric, ordered coefficient tree, not a function-
intrinsic polynomial or rational quotient. No compiler or production imports.
Symmetric-coordinate QR is our shape-only extension for deficient completions.
"""

import numpy as np


def _real(value, ndim):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError("real arrays required")
    result = np.array(raw, dtype=np.float64, copy=True)
    if result.ndim != ndim or 0 in result.shape or not np.isfinite(result).all():
        raise ValueError(f"finite nonempty {ndim}-dimensional array required")
    return result


def rq(matrix):
    """Right-isometric QR of the transpose, with a lower-triangular left factor.

    This is the LQ convention, not an upper-triangular RQ gauge. Retain all
    shape-selected Q columns. Either triangular orientation serves Algorithm 1.
    """
    q, r = np.linalg.qr(_real(matrix, 2).T, mode="reduced")
    return _real(r.T, 2), _real(q.T, 2)


def _symmetric(core):
    core = _real(core, 3)
    if core.shape[1] != core.shape[2]:
        raise ValueError("a tied core must have equal input dimensions")
    scale = max(float(np.max(np.abs(core))), np.finfo(float).tiny)
    if np.max(np.abs(core - core.swapaxes(1, 2))) > 1e-12 * scale:
        raise ValueError("symmetrize the input core before ODT")
    return core / 2 + core.swapaxes(1, 2) / 2


def symmetric_rq(core):
    """Direct RQ in an orthonormal coordinate basis of symmetric matrices.

    Store C_ii and sqrt(2)*C_ij (i<j), so null-space completions stay symmetric.
    Width is min(out, in*(in+1)//2), chosen from shape, never numerical rank.
    """
    core = _symmetric(core)
    i, j = np.triu_indices(core.shape[1])
    weights = np.where(i == j, 1., np.sqrt(2.))
    r, packed = rq(core[:, i, j] * weights)
    q = np.zeros((packed.shape[0], core.shape[1], core.shape[2]))
    q[:, i, j] = packed / weights
    q[:, j, i] = packed / weights
    return r, q


def orthogonalize(embedding, cores, head):
    """Algorithm 1: direct RQ, pushing R into BOTH next-core input legs."""
    embedding, head = _real(embedding, 2), _real(head, 2)
    cores = tuple(_symmetric(core) for core in cores)
    width = embedding.shape[0]
    for core in cores:
        if core.shape[1:] != (width, width):
            raise ValueError("chain input dimensions disagree")
        width = core.shape[0]
    if head.shape[1] != width:
        raise ValueError("head input dimension disagrees")
    r, embedding = rq(embedding)
    canonical = []
    for core in cores:
        transformed = np.einsum("oij,ia,jb->oab", core, r, r)
        r, q = symmetric_rq(transformed)
        canonical.append(q)
    return embedding, tuple(canonical), _real(head @ r, 2)


def environments(cores, head):
    """Algorithm 2, ONE occurrence per layer, after orthogonalize.

    Contract the downstream network, explicitly tracing the sibling leg.
    The caller must supply canonical cores. This is not a raw-weight metric.
    """
    head = _real(head, 2)
    environment = _real(np.einsum("oa,ob->ab", head, head), 2)
    result = [environment]
    for core in reversed(cores):
        core = _real(core, 3)
        environment = _real(np.einsum("oia,op,pja->ij", core, environment, core), 2)
        result.append(environment)
    return tuple(reversed(result))


def eigenspaces(envs, ranks=None):
    """Leading environment eigenvectors; a cut through a tie is non-unique."""
    envs = tuple(_real(env, 2) for env in envs)
    ranks = tuple(env.shape[0] for env in envs) if ranks is None else tuple(ranks)
    if len(ranks) != len(envs):
        raise ValueError("one retained dimension required per bond")
    bases = []
    for env, rank in zip(envs, ranks):
        if env.shape[0] != env.shape[1]:
            raise ValueError("environment must be square")
        if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= len(env):
            raise ValueError("retained dimension must be an integer in [1,width]")
        scale = max(float(np.max(np.abs(env))), np.finfo(float).tiny)
        if np.max(np.abs(env - env.T)) > 1e-12 * scale:
            raise ValueError("environment is not symmetric")
        values, vectors = np.linalg.eigh(env / 2 + env.T / 2)
        if not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise ValueError("environment eigendecomposition exceeded finite arithmetic")
        if values[0] < -1e-12 * len(env) * scale:
            raise ValueError("environment is materially indefinite")
        bases.append(vectors[:, np.argsort(values)[::-1]][:, :rank])
    return tuple(bases)


def absorb(embedding, cores, head, bases):
    """Algorithm 3: contract each retained basis into every adjacent leg.

    Bases come from eigenspaces. Full width changes coordinates only; reduced
    width physically slices the tree. Orthogonality is a caller precondition.
    """
    cores, bases = tuple(cores), tuple(bases)
    if len(bases) != len(cores) + 1:
        raise ValueError("one basis required at every hidden bond")
    reduced = tuple(np.einsum("ou,oij,ia,jb->uab", bases[k + 1], core,
                             bases[k], bases[k]) for k, core in enumerate(cores))
    return bases[0].T @ embedding, reduced, head @ bases[-1]
