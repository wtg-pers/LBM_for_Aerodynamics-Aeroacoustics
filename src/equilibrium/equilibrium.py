class Maxwellian:
    def __init__(self, xp, lattice, domain):
        self.xp = xp
        self.lattice = lattice
    
    def compute(self, rho, u):
        xp = self.xp
        c = self.lattice.c
        w = self.lattice.w

        usqr = xp.sum(u**2, axis=0)
        cu = xp.einsum('dq,d...-> q...', c, u)

        # Broadcast weights to (Q, 1, 1[, 1]) depending on rho.ndim
        w_bc = w.astype(rho.dtype).reshape((-1,) + (1,) * rho.ndim)

        return w_bc * rho * (1.0 + 3.0 * cu + 4.5 * (cu**2) - 1.5 * usqr)
