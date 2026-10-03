"""
Attack cost law for STA-1.

    z = (p - gamma) * sqrt(T) / sqrt(gamma(1-gamma))

Watermark strength grows as sqrt(T) while the document grows as
T, so the attacker's working space outpaces the signal.

Edits required to reach a target z:

    E = [(p-gamma)T - z* sqrt(gamma(1-gamma)T)] / eta

eta is greens destroyed per edit, bounded above by 2 since a
substitution touches two adjacent pairs.
"""

import math

GAMMA = 0.5
Z_THRESHOLD = 2.0


def z_score(p, T, gamma=GAMMA):
    """Detector z from green fraction and pair count."""
    if T <= 0:
        return 0.0
    return (p - gamma) * T / math.sqrt(gamma * (1 - gamma) * T)


def greens_to_remove(p, T, z_target=Z_THRESHOLD, gamma=GAMMA):
    """
    Green pairs that must be destroyed to reach z_target.

    Negative means the document already sits below target and
    needs no attack.
    """
    present = (p - gamma) * T
    tolerated = z_target * math.sqrt(gamma * (1 - gamma) * T)
    return present - tolerated


def edits_required(p, T, eta, z_target=Z_THRESHOLD, gamma=GAMMA):
    """Edits needed, given eta greens destroyed per edit."""
    need = greens_to_remove(p, T, z_target, gamma)
    if need <= 0:
        return 0.0
    return need / eta


def eta_observed(p, T, edits, z_target=Z_THRESHOLD, gamma=GAMMA):
    """
    Recover eta from a completed attack.

    Inverts edits_required, letting us measure efficiency from
    real runs instead of assuming it.
    """
    need = greens_to_remove(p, T, z_target, gamma)
    return need / edits if edits > 0 else None

