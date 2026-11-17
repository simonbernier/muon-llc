"""Singular-learning-theory observables shared by the testbeds.

Samplers for the localised tempered posterior p(w) ∝ exp(-nβ L_n(w) - γ/2 ||w - w0||²) (SGLD,
MALA on a β-ladder) and the estimators of the local learning coefficient built on them (WBIC
form, ladder slope fit with multiplicity, thermodynamic integration). They act on any object
satisfying `slt.problem.Problem`.
"""
