import math
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms as rdMT


####
# This script contains a set of molecule augmentations meant to run 
# on cpu, during dataloading, and using rdkit 
####

try:
    from rdkit.Chem import rdDetermineBonds as rddb
    _HAS_DETERMINE_BONDS = True
except Exception:
    _HAS_DETERMINE_BONDS = False

def center(data):
    # Center the molecular coordinates
    data.pos = data.pos - data.pos.mean(dim=0, keepdim=True)
    return data

def invert(data, prob = 0.5):
    # Invert the molecular coordinates with probability prob
    if torch.rand(1).item() < prob:
        data.pos = -data.pos
    return data

def random_rotate(data, 
                  max_angle : float=2*math.pi):
    R = get_rand_rot(max_angle, device=data.pos.device, dtype=data.pos.dtype)
    data.pos = data.pos @ R.T  # Apply rotation
    return data

def get_rand_rot(max_angle : float = 2*math.pi, device=None, dtype=None) -> torch.Tensor:
    '''
    max angle given in radians

    returns:
        3x3 rotation matrix
    '''
    angles = (torch.rand(3, device=device) * 2 - 1) * max_angle
    cx, cy, cz = torch.cos(angles)
    sx, sy, sz = torch.sin(angles)
    # Rotation matrices for X, Y, Z axes
    Rx = torch.tensor([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx,  cx]], dtype=dtype, device=device)

    Ry = torch.tensor([[ cy, 0, sy],
                       [  0, 1,  0],
                       [-sy, 0, cy]], dtype=dtype, device=device)

    Rz = torch.tensor([[cz, -sz, 0],
                       [sz,  cz, 0],
                       [ 0,   0, 1]], dtype=dtype, device=device)

    # Combined rotation (Z-Y-X intrinsic order)
    R = Rz @ Ry @ Rx  # [3, 3]
    return R

def torsion_transform(data, # torch geometric data object
                      angle_std = 18.0,
                      max_torsions = 10,
                      ):
    
    #Randomly rotate a dihedral angles
    new_pos, tor_edges, tor_angles = rotate_random_dihedrals(
                                        data.z, 
                                        data.pos, 
                                        v_deg_max=angle_std, 
                                        k=max_torsions, 
                                        seed=None,
                                        strict_rotatable=True, 
                                        total_charge=None,
                                        distribution="gaussian",
                                        output_edges=True,
                                        return_indices=True,
                                        )
    
    data.pos = new_pos
    data.tor_edge_index = tor_edges #key must have 'edge_index' or 'adj' in name to colate properly
    data.tor_angles = tor_angles
    
    return data 



def _connect_by_distance(mol: Chem.Mol, scale: float = 1.20, min_dist: float = 0.4):
    """
    Add single bonds based on covalent radii and interatomic distances.
    scale   : multiplier on (r_cov[i] + r_cov[j]) threshold
    min_dist: ignore absurdly short distances
    """
    N = mol.GetNumAtoms()
    conf = mol.GetConformer()
    pt = Chem.GetPeriodicTable()

    Z = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(N)], dtype=int)
    R = np.array([pt.GetRcovalent(int(z)) or 0.77 for z in Z], dtype=float)  # Å; fallback ~H

    xyz = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(N)], dtype=float)

    # pairwise distances
    diff = xyz[:, None, :] - xyz[None, :, :]
    D = np.linalg.norm(diff, axis=-1)

    thresh = scale * (R[:, None] + R[None, :])

    rw = Chem.RWMol(mol)
    for i in range(N):
        for j in range(i+1, N):
            if rw.GetBondBetweenAtoms(i, j) is not None:
                continue
            dij = D[i, j]
            if dij < min_dist:
                continue
            if np.isfinite(thresh[i, j]) and dij < thresh[i, j]:
                rw.AddBond(i, j, Chem.BondType.SINGLE)

    mol2 = rw.GetMol()
    # ring perception so Bond.IsInRing() works
    Chem.GetSymmSSSR(mol2)
    return mol2

def _perceive_bonds(mol: Chem.Mol, total_charge: int | None):
    """
    Prefer DetermineBonds if available & a charge is provided; otherwise
    fall back to distance-based connectivity that doesn't need charge.
    """
    if _HAS_DETERMINE_BONDS and total_charge is not None:
        try:
            # Some RDKit versions require positional argument
            rddb.DetermineBonds(mol, int(total_charge))
            return mol
        except Exception:
            pass  # fall through to distance method
    return _connect_by_distance(mol)

def rotate_random_dihedral(
    z: torch.Tensor,                 # [N,1] int atomic numbers
    xyz: torch.Tensor,               # [N,3] float Å
    v_deg: float,                    # rotation increment in degrees
    seed: int | None = None,
    strict_rotatable: bool = True,
    total_charge: int | None = None, # pass if you know net charge; else leave None
    return_indices: bool = False,
):
    # assert z.dim() == 2 and z.size(1) == 1, "z must be [N,1]"
    assert xyz.dim() == 2 and xyz.size(1) == 3, "xyz must be [N,3]"

    device, dtype = xyz.device, xyz.dtype
    N = xyz.size(0)
    if N < 4:
        return (xyz.clone(), None) if return_indices else xyz.clone()

    rng = np.random.default_rng(seed)

    # Build geometry-only molecule with conformer
    rw = Chem.RWMol()
    for aZ in z.view(-1).tolist():
        rw.AddAtom(Chem.Atom(int(aZ)))
    mol = rw.GetMol()

    conf = Chem.Conformer(N)
    xyz_np = xyz.detach().cpu().numpy()
    for i in range(N):
        x, y, zf = map(float, xyz_np[i])
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, zf))
    mol.AddConformer(conf, assignId=True)

    # Connectivity
    mol = _perceive_bonds(mol, total_charge=total_charge)

    # Candidate rotatable bonds
    candidates: list[tuple[int,int]] = []
    for b in mol.GetBonds():
        if b.IsInRing():
            continue
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a = b.GetBeginAtom()
        c = b.GetEndAtom()
        if a.GetDegree() <= 1 or c.GetDegree() <= 1:
            continue
        if strict_rotatable:
            if b.GetIsConjugated():
                continue
            if a.GetAtomicNum() == 1 or c.GetAtomicNum() == 1:
                continue
        candidates.append((a.GetIdx(), c.GetIdx()))

    if not candidates:
        return (xyz.clone(), None) if return_indices else xyz.clone()

    j, k = candidates[rng.integers(len(candidates))]

    # Pick neighbors i (of j) and l (of k) to define i–j–k–l
    j_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
    k_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]

    tries = 0
    while (not j_nbrs or not k_nbrs) and tries < 16 and len(candidates) > 1:
        j, k = candidates[rng.integers(len(candidates))]
        j_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
        k_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
        tries += 1

    if not j_nbrs or not k_nbrs:
        return (xyz.clone(), None) if return_indices else xyz.clone()

    i = int(rng.choice(j_nbrs))
    l = int(rng.choice(k_nbrs))

    # Rotate the dihedral
    conf = mol.GetConformer()
    cur = rdMT.GetDihedralDeg(conf, i, j, k, l)
    rdMT.SetDihedralDeg(conf, i, j, k, l, float(cur + v_deg))

    # Fetch updated coordinates back to torch
    new_xyz = torch.empty_like(xyz)
    for idx in range(N):
        p = conf.GetAtomPosition(idx)
        new_xyz[idx, 0], new_xyz[idx, 1], new_xyz[idx, 2] = p.x, p.y, p.z
    new_xyz = new_xyz.to(device=device, dtype=dtype)

    return (new_xyz, (i, j, k, l)) if return_indices else new_xyz




# def rotate_random_dihedrals(
#     z: torch.Tensor,                 # [N,1] atomic numbers (int)
#     xyz: torch.Tensor,               # [N,3] coordinates in Å (float)
#     v_deg_max: float,                # max absolute rotation per dihedral (degrees)
#     k: int,                          # rotate up to k distinct dihedrals
#     *,
#     seed: int | None = None,
#     strict_rotatable: bool = True,   # avoid conjugated and X–H torsions
#     total_charge: int | None = None, # pass if you know net charge; else None
#     return_indices: bool = False,    # return list of (i,j,k,l,Δ) actually rotated
#     distribution: str = "uniform"    # "uniform" or "gaussian"
# ):
#     """
#     Rotate up to k random dihedrals by random angles in [-v_deg_max, +v_deg_max].
#     Returns new coordinates (torch, same dtype/device) and optionally the list
#     of rotated dihedrals with applied angles.

#     Notes
#     -----
#     - Bonding is inferred from 3D geometry (no SMILES needed).
#     - Each dihedral rotation moves the entire fragment on the 'l' side about j–k.
#     - If fewer than k rotatable bonds exist, rotates as many as available.
#     """
#     assert z.dim() == 2 and z.size(1) == 1, "z must be [N,1]"
#     assert xyz.dim() == 2 and xyz.size(1) == 3, "xyz must be [N,3]"
#     device, dtype = xyz.device, xyz.dtype
#     N = xyz.size(0)
#     if N < 4 or k <= 0 or v_deg_max <= 0:
#         return (xyz.clone(), []) if return_indices else xyz.clone()

#     rng = np.random.default_rng(seed)

#     # ---- build an RDKit molecule with conformer from (z, xyz) ----
#     rw = Chem.RWMol()
#     for aZ in z.view(-1).tolist():
#         rw.AddAtom(Chem.Atom(int(aZ)))
#     mol = rw.GetMol()
#     conf = Chem.Conformer(N)
#     xyz_np = xyz.detach().cpu().numpy()
#     for i in range(N):
#         x, y, zf = map(float, xyz_np[i])
#         conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, zf))
#     mol.AddConformer(conf, assignId=True)

#     # ---- connectivity (robust) ----
#     mol = _perceive_bonds(mol, total_charge=total_charge)

#     # ---- gather rotatable bond candidates (non-ring, single, non-terminal) ----
#     candidates: list[tuple[int, int]] = []
#     for b in mol.GetBonds():
#         if b.IsInRing():
#             continue
#         if b.GetBondType() != Chem.BondType.SINGLE:
#             continue
#         a = b.GetBeginAtom(); c = b.GetEndAtom()
#         if a.GetDegree() <= 1 or c.GetDegree() <= 1:
#             continue
#         if strict_rotatable:
#             if b.GetIsConjugated():
#                 continue
#             if a.GetAtomicNum() == 1 or c.GetAtomicNum() == 1:
#                 continue
#         candidates.append((a.GetIdx(), c.GetIdx()))

#     if not candidates:
#         return (xyz.clone(), []) if return_indices else xyz.clone()

#     # sample up to k bonds without replacement
#     n_pick = min(k, len(candidates))
#     pick_idx = rng.choice(len(candidates), size=n_pick, replace=False)
#     picked_bonds = [candidates[i] for i in pick_idx.tolist()]

#     # For each picked bond j–k, choose neighbors i (≠k) and l (≠j) to define i–j–k–l
#     rotated: list[tuple[int, int, int, int, float]] = []
#     conf = mol.GetConformer()

#     for (j, k_) in picked_bonds:
#         j_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k_]
#         k_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(k_).GetNeighbors() if n.GetIdx() != j]
#         if not j_nbrs or not k_nbrs:
#             continue  # skip degenerate ones

#         i = int(rng.choice(j_nbrs))
#         l = int(rng.choice(k_nbrs))

#         if distribution == "uniform":
#             delta = float(rng.uniform(-v_deg_max, +v_deg_max))  # random signed angle
#         elif distribution == "gaussian":
#             delta = float(rng.normal(0, v_deg_max))  # random signed angle

#         current = rdMT.GetDihedralDeg(conf, i, j, k_, l)
#         rdMT.SetDihedralDeg(conf, i, j, k_, l, current + delta)
#         rotated.append((i, j, k_, l, delta))

#     # If nothing rotated (e.g., all degenerate), return original
#     if not rotated:
#         return (xyz.clone(), []) if return_indices else xyz.clone()

#     # ---- fetch updated coordinates back to torch ----
#     new_xyz = torch.empty_like(xyz)
#     for idx in range(N):
#         p = conf.GetAtomPosition(idx)
#         new_xyz[idx, 0], new_xyz[idx, 1], new_xyz[idx, 2] = p.x, p.y, p.z
#     new_xyz = new_xyz.to(device=device, dtype=dtype)

#     return (new_xyz, rotated) if return_indices else new_xyz




def rotate_random_dihedrals(
    z: torch.Tensor,                 # [N,1] atomic numbers (int)
    xyz: torch.Tensor,               # [N,3] coordinates in Å (float)
    v_deg_max: float,                # max |Δ| per dihedral (degrees)
    k: int,                          # rotate up to k distinct dihedrals
    *,
    seed: int | None = None,
    distribution: str = "gaussian",  # "uniform" or "gaussian"
    deg_upperbound: float = 55.0,    # when using gaussian distribution angles equal and above this value will be zeroed out
    strict_rotatable: bool = True,   # avoid conjugated and X–H torsions
    total_charge: int | None = None, # pass if known; else None
    return_indices: bool = False,    # (ignored if output_edges=True)
    output_edges: bool = False,      # <-- NEW: return (xyz, edges, torsion_deltas)
):
    """
    Rotate up to k random dihedrals by random angles in [-v_deg_max, +v_deg_max].

    Returns
    -------
    If output_edges == False:
        xyz_new : [N,3]  (and optionally the list of rotated torsions if return_indices=True)
    If output_edges == True:
        xyz_new : [N,3]
        edges   : [2,E]  (directed edges; both directions for every bond)
        torsion_delta_deg : [E] (per-edge applied rotation in degrees; zeros if none)
    """
    # assert z.dim() == 2 and z.size(1) == 1, "z must be [N,1]"
    assert xyz.dim() == 2 and xyz.size(1) == 3, "xyz must be [N,3]"
    device, dtype = xyz.device, xyz.dtype
    N = xyz.size(0)
    if N < 4 or k <= 0 or v_deg_max <= 0:
        if output_edges:
            # Build trivial edge list (if any bonds exist) before returning
            # Construct a minimal mol to perceive bonds for edges
            rw = Chem.RWMol()
            for aZ in z.view(-1).tolist():
                rw.AddAtom(Chem.Atom(int(aZ)))
            mol = rw.GetMol()
            conf = Chem.Conformer(N)
            xyz_np = xyz.detach().cpu().numpy()
            for i in range(N):
                x, y, zf = map(float, xyz_np[i])
                conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, zf))
            mol.AddConformer(conf, assignId=True)
            mol = _perceive_bonds(mol, total_charge=total_charge)
            bonds = list(mol.GetBonds())
            B = len(bonds)
            edges = torch.empty((2, 2*B), dtype=torch.long, device=device)
            deltas = torch.zeros(2*B, dtype=dtype, device=device)
            for e, b in enumerate(bonds):
                a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                edges[:, 2*e]   = torch.tensor([a, c], device=device)
                edges[:, 2*e+1] = torch.tensor([c, a], device=device)
            return xyz.clone(), edges, deltas
        else:
            return (xyz.clone(), []) if return_indices else xyz.clone()

    rng = np.random.default_rng(seed)

    # ---- build RDKit molecule with conformer from (z, xyz) ----
    rw = Chem.RWMol()
    for aZ in z.view(-1).tolist():
        rw.AddAtom(Chem.Atom(int(aZ)))
    mol = rw.GetMol()

    conf = Chem.Conformer(N)
    xyz_np = xyz.detach().cpu().numpy()
    for i in range(N):
        x, y, zf = map(float, xyz_np[i])
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, zf))
    mol.AddConformer(conf, assignId=True)

    # ---- connectivity ----
    mol = _perceive_bonds(mol, total_charge=total_charge)

    # ---- gather rotatable bond candidates ----
    candidates: list[tuple[int, int]] = []
    for b in mol.GetBonds():
        if b.IsInRing():
            continue
        if b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a = b.GetBeginAtom()
        c = b.GetEndAtom()
        if a.GetDegree() <= 1 or c.GetDegree() <= 1:  # avoid terminal
            continue
        if strict_rotatable:
            if b.GetIsConjugated():
                continue
            if a.GetAtomicNum() == 1 or c.GetAtomicNum() == 1:
                continue
        candidates.append((a.GetIdx(), c.GetIdx()))

    # sample as many as possible
    n_pick = min(k, len(candidates))
    if n_pick == 0:
        # still may want edges/deltas
        if output_edges:
            bonds = list(mol.GetBonds())
            B = len(bonds)
            edges = torch.empty((2, 2*B), dtype=torch.long, device=device)
            deltas = torch.zeros(2*B, dtype=dtype, device=device)
            for e, b in enumerate(bonds):
                a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                edges[:, 2*e]   = torch.tensor([a, c], device=device)
                edges[:, 2*e+1] = torch.tensor([c, a], device=device)
            return xyz.clone(), edges, deltas
        else:
            return (xyz.clone(), []) if return_indices else xyz.clone()

    pick_idx = rng.choice(len(candidates), size=n_pick, replace=False)
    picked_bonds = [candidates[i] for i in pick_idx.tolist()]

    # ---- rotate each chosen dihedral once ----
    rotated: list[tuple[int, int, int, int, float]] = []
    conf = mol.GetConformer()

    for (j, k_) in picked_bonds:
        j_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k_]
        k_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(k_).GetNeighbors() if n.GetIdx() != j]
        if not j_nbrs or not k_nbrs:
            continue

        i = int(rng.choice(j_nbrs))
        l = int(rng.choice(k_nbrs))

        # delta = float(rng.uniform(-v_deg_max, +v_deg_max))  # degrees
        if distribution == "uniform":
            delta = float(rng.uniform(-v_deg_max, +v_deg_max))  # random signed angle
        elif distribution == "gaussian":
            delta = float(rng.normal(0, v_deg_max))  # random signed angle

            #zero out deltas that are too large
            if abs(delta) >= deg_upperbound:
                delta = 0.0

        current = rdMT.GetDihedralDeg(conf, i, j, k_, l)
        rdMT.SetDihedralDeg(conf, i, j, k_, l, current + delta)
        rotated.append((i, j, k_, l, delta))

    # ---- fetch updated coordinates back to torch ----
    new_xyz = torch.empty_like(xyz)
    for idx in range(N):
        p = conf.GetAtomPosition(idx)
        new_xyz[idx, 0], new_xyz[idx, 1], new_xyz[idx, 2] = p.x, p.y, p.z
    new_xyz = new_xyz.to(device=device, dtype=dtype)

    if not output_edges:
        return (new_xyz, rotated) if return_indices else new_xyz

    # ---- Build directed edge list and per-edge torsion deltas ----
    bonds = list(mol.GetBonds())
    B = len(bonds)
    E = 2 * B

    edges = torch.empty((2, E), dtype=torch.long, device=device)
    torsion_delta_deg = torch.zeros(E, dtype=dtype, device=device)

    # map directed bond -> delta (degrees); j->k = +Δ, k->j = -Δ
    dir_delta = {}
    for (i, j, k_, l, delta) in rotated:
        dir_delta[(j, k_)] = dir_delta.get((j, k_), 0.0) + delta
        dir_delta[(k_, j)] = dir_delta.get((k_, j), 0.0) - delta

    for e, b in enumerate(bonds):
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        # forward edge
        edges[0, 2*e]   = a
        edges[1, 2*e]   = c
        torsion_delta_deg[2*e] = float(dir_delta.get((a, c), 0.0))
        # reverse edge
        edges[0, 2*e+1] = c
        edges[1, 2*e+1] = a
        torsion_delta_deg[2*e+1] = float(dir_delta.get((c, a), 0.0))

    return new_xyz, edges, torsion_delta_deg















############################################ SCRATCH

# import math
# import torch
# import numpy as np
# from rdkit import Chem
# from rdkit.Chem import rdMolTransforms as rdMT

# # RDKit's distance-based bond perception
# try:
#     from rdkit.Chem import rdDetermineBonds as rddb
#     _HAS_DETERMINE_BONDS = True
# except Exception:
#     _HAS_DETERMINE_BONDS = False


# def rand_dihedral_rot(z, pos, max_angle):
#     # Randomly select a dihedral angle
#     angle = np.random.uniform(-max_angle, max_angle)
#     # Apply the rotation to the molecule
#     return rotate_random_dihedral(z, pos, angle, seed=None, return_indices=False)


# def rotate_random_dihedral(
#     z: torch.Tensor,                 # [N,1] int (atomic numbers)
#     xyz: torch.Tensor,               # [N,3] float (Å)
#     v_deg: float,                    # rotation increment in degrees
#     seed: int | None = None,
#     strict_rotatable: bool = True,   # exclude conjugated/amide-like bonds
#     return_indices: bool = False,    # optionally return (i,j,k,l) picked
# ) -> tuple[torch.Tensor, tuple[int,int,int,int] | None]:
#     """
#     Rotate a random dihedral by v_deg (degrees) and return new coordinates.

#     Notes
#     -----
#     * Bonds are inferred from 3D geometry (covalent radii) via rdDetermineBonds.
#     * The rotation moves the entire fragment on the side of atom 'l' around the j–k axis.
#     * If no suitable rotatable dihedral is found, the original coordinates are returned.

#     Returns
#     -------
#     xyz_new : torch.Tensor [N,3]
#     (i,j,k,l) : optional tuple of indices (if return_indices=True), else None
#     """
#     if z.dim() == 1:
#         z = z.unsqueeze(1)
#     assert z.dim() == 2 and z.size(1) == 1, "z must be [N,1]"
#     assert xyz.dim() == 2 and xyz.size(1) == 3, "xyz must be [N,3]"

#     device, dtype = xyz.device, xyz.dtype
#     N = xyz.size(0)
#     if N < 4:
#         return xyz.clone(), None if return_indices else (xyz.clone())

#     # RNG
#     rng = np.random.default_rng(seed)

#     # ---- build an RDKit molecule with a conformer ----
#     rw = Chem.RWMol()
#     for aZ in z.view(-1).tolist():
#         rw.AddAtom(Chem.Atom(int(aZ)))

#     mol = rw.GetMol()
#     conf = Chem.Conformer(N)
#     xyz_np = xyz.detach().cpu().numpy()
#     for i in range(N):
#         x, y, zf = map(float, xyz_np[i])
#         conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, zf))
#     mol.AddConformer(conf, assignId=True)

#     # ---- perceive bonds from geometry ----
#     if not _HAS_DETERMINE_BONDS:
#         raise RuntimeError(
#             "rdDetermineBonds not available in this RDKit build. "
#             "Update RDKit (>= 2020.x) or prebuild bonds yourself."
#         )
#     # This modifies the molecule in place (adds bonds based on distances).
#     rddb.DetermineBonds(mol, charge=0)

#     # ---- collect candidate rotatable bonds ----
#     candidates: list[tuple[int,int]] = []
#     for b in mol.GetBonds():
#         if b.IsInRing():  # avoid ring torsions
#             continue
#         if b.GetBondType() != Chem.BondType.SINGLE:
#             continue
#         a = b.GetBeginAtom()
#         c = b.GetEndAtom()
#         # exclude terminal bonds
#         if a.GetDegree() <= 1 or c.GetDegree() <= 1:
#             continue
#         if strict_rotatable:
#             # exclude conjugated / amide-like single bonds
#             if b.GetIsConjugated():
#                 continue
#             # also exclude bonds to metals (optional)
#             if a.GetAtomicNum() == 1 or c.GetAtomicNum() == 1:
#                 # H-X torsions are ill-defined for fragment rotation
#                 continue
#         candidates.append((a.GetIdx(), c.GetIdx()))

#     if not candidates:
#         # nothing to rotate
#         return xyz.clone(), None if return_indices else (xyz.clone())

#     j, k = candidates[rng.integers(len(candidates))]

#     # pick neighbors i (of j, not k) and l (of k, not j)
#     j_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
#     k_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]

#     # if any side is ambiguous, try other bonds a few times
#     trials = 0
#     while (not j_nbrs or not k_nbrs) and trials < 16 and len(candidates) > 1:
#         j, k = candidates[rng.integers(len(candidates))]
#         j_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(j).GetNeighbors() if n.GetIdx() != k]
#         k_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(k).GetNeighbors() if n.GetIdx() != j]
#         trials += 1

#     if not j_nbrs or not k_nbrs:
#         # fallback: no torsion with four distinct atoms found
#         return xyz.clone(), None if return_indices else (xyz.clone())

#     i = int(rng.choice(j_nbrs))
#     l = int(rng.choice(k_nbrs))

#     # ---- rotate dihedral by v_deg ----
#     conf = mol.GetConformer()
#     current = rdMT.GetDihedralDeg(conf, i, j, k, l)
#     rdMT.SetDihedralDeg(conf, i, j, k, l, float(current + v_deg))

#     # ---- fetch updated coordinates back to torch ----
#     new_xyz = torch.empty_like(xyz)
#     for idx in range(N):
#         p = conf.GetAtomPosition(idx)
#         new_xyz[idx, 0] = p.x
#         new_xyz[idx, 1] = p.y
#         new_xyz[idx, 2] = p.z

#     new_xyz = new_xyz.to(device=device, dtype=dtype)

#     if return_indices:
#         return new_xyz, (i, j, k, l)
#     else:
#         return new_xyz