"""Wall-model closures for under-resolved boundary layers.

PLACEHOLDER — populated in the wall-model phase. Planned closures:

    - Equilibrium wall function (log-law iteration)
    - Werner-Wengle power-law
    - Non-equilibrium two-layer (Cabot, Wang)

The closure here is the physics piece (compute u_tau / nu_t_wall from the
off-wall velocity); the kernel-side application of tau_w lives in
src/boundary/.
"""
