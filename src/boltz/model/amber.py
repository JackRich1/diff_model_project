import torch

JOULES_PER_CALORIE = 4.184

def get_bond_energy(pos, bond_index, k, r_0):
    """
    Computes the harmonic bond energy and force given by
    E = k * (r - r_0) ** 2
        k is the force constant
        r_0 is the equilibrium bond length
    """
    r_ij = pos.index_select(-2, bond_index[0]) -  pos.index_select(-2, bond_index[1])
    r_ij_norm = torch.linalg.norm(r_ij, dim=-1)
    r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)

    bond_energy = (k * (r_ij_norm - r_0) ** 2).sum(axis=-1)
    bond_energy = bond_energy * JOULES_PER_CALORIE
    return bond_energy

def get_angle_energy(pos, angle_index, k, theta_0):
    """
    Computes the harmonic angle energy and force given by
    E = k * (theta - theta_0) ** 2
        k is the force constant
        theta_0 is the equilibrium bond angle
    """
    r_ij = pos.index_select(-2, angle_index[0]) - pos.index_select(-2, angle_index[1])
    r_kj = pos.index_select(-2, angle_index[2]) - pos.index_select(-2, angle_index[1])
    
    r_ij_norm = torch.linalg.norm(r_ij, axis=-1)
    r_kj_norm = torch.linalg.norm(r_kj, axis=-1)
    
    r_hat_ij = r_ij / r_ij_norm.unsqueeze(-1)
    r_hat_kj = r_kj / r_kj_norm.unsqueeze(-1)
    
    cos_theta = (r_hat_ij.unsqueeze(-2) @ r_hat_kj.unsqueeze(-1)).squeeze(-1, -2)
    theta = torch.arccos(cos_theta)
    angle_energy = (k * (theta - theta_0) ** 2).sum(axis=-1)
    angle_energy = angle_energy * JOULES_PER_CALORIE
    return angle_energy

def get_torsion_energy(pos, torsion_index, k, n, phi_0):
    """
    Computes the periodic torsion energy and force given by
    E = k * (1 + cos(n * phi - phi_0))
        k is the force constant
        n is the periodicity
        phi_0 is the phase offset
    """
    r_ij = pos.index_select(-2, torsion_index[0]) - pos.index_select(-2, torsion_index[1])
    r_kj = pos.index_select(-2, torsion_index[2]) - pos.index_select(-2, torsion_index[1])
    r_kl = pos.index_select(-2, torsion_index[2]) - pos.index_select(-2, torsion_index[3])

    n_ijk = torch.cross(r_ij, r_kj, dim=-1)
    n_jkl = torch.cross(r_kj, r_kl, dim=-1)
    
    r_kj_norm = torch.linalg.norm(r_kj, dim=-1)
    n_ijk_norm = torch.linalg.norm(n_ijk, dim=-1)
    n_jkl_norm = torch.linalg.norm(n_jkl, dim=-1)
    
    sign_phi = torch.sign(r_kj.unsqueeze(-2) @ torch.cross(n_ijk, n_jkl).unsqueeze(-1)).squeeze(-1, -2)
    phi = sign_phi * torch.arccos(
        torch.clamp(
            (n_ijk.unsqueeze(-2) @ n_jkl.unsqueeze(-1)).squeeze(-1,-2) /
            (n_ijk_norm * n_jkl_norm),
            -1 + 1e-7,
            1 - 1e-7
        )
    )

    torsion_energy = (k.unsqueeze(0) * (1 + torch.cos(n * phi - phi_0.unsqueeze(0)))).sum(axis=-1)
    torsion_energy = torsion_energy * JOULES_PER_CALORIE
    return torsion_energy

def compute_bonded_energy_decomp(pred_atom_coords, feats):
    bond_energy = get_bond_energy(
        pred_atom_coords,
        feats['amber_bond_index'][0],
        feats['amber_bond_k'][0],
        feats['amber_bond_r_eq'][0],
    )
    angle_energy = get_angle_energy(
        pred_atom_coords,
        feats['amber_angle_index'][0],
        feats['amber_angle_k'][0],
        feats['amber_angle_theta_eq'][0] * torch.pi / 180,
    )
    torsion_energy = get_torsion_energy(
        pred_atom_coords,
        feats['amber_torsion_index'][0],
        feats['amber_torsion_k'][0],
        feats['amber_torsion_n'][0],
        feats['amber_torsion_phi'][0] * torch.pi / 180,
    )
    bonded_energy = bond_energy + angle_energy + torsion_energy
    out = {
        'bond': bond_energy.cpu().numpy(),
        'angle': angle_energy.cpu().numpy(),
        'torsion': torsion_energy.cpu().numpy(),
        'bonded': bonded_energy.cpu().numpy()
    }
    return out