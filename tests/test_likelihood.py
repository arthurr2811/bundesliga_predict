import numpy as np

from bundesliga_predict.model.likelihood import tau_correction


def test_tau_only_touches_low_scores():
    goals = np.arange(6)
    home, away = np.meshgrid(goals, goals, indexing="ij")

    tau = tau_correction(home, away, 1.5, 1.2, rho=-0.1)

    untouched = np.ones_like(tau, dtype=bool)
    untouched[:2, :2] = False
    assert np.allclose(tau[untouched], 1.0)
    assert not np.allclose(tau[:2, :2], 1.0)


def test_tau_is_neutral_for_rho_zero():
    goals = np.arange(4)
    home, away = np.meshgrid(goals, goals, indexing="ij")

    assert np.allclose(tau_correction(home, away, 1.5, 1.2, rho=0.0), 1.0)


def test_tau_shifts_mass_as_dixon_coles_intend():
    # Negatives rho: mehr Unentschieden (0:0, 1:1), weniger knappe Fuehrungen.
    tau = tau_correction(
        np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]), 1.5, 1.2, rho=-0.1
    )
    assert tau[0] > 1.0 and tau[1] > 1.0
    assert tau[2] < 1.0 and tau[3] < 1.0
