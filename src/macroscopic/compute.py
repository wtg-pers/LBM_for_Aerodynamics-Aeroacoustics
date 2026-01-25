class Macroscopic:
    def __init__(self, xp, lattice):
        self.xp = xp
        self.lattice = lattice
        self.c = lattice.c

    def compute(self, f):
        rho = self.xp.sum(f, axis=0)
        # c: (3, Q), f: (Q, ...) -> u: (3, ...)
        momentum = self.xp.tensordot(self.c, f, axes=(1, 0)) 
        u = momentum / rho[None, ...]
        
        return rho, u