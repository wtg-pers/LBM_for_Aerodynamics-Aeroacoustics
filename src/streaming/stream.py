class STREAMROLL:
    def __init__(self, xp, lattice):
        self.xp = xp
        self.lattice = lattice
        self.c = xp.asarray(lattice.c)  # (3, Q)

    def compute(self, f_post, f_next):
        # f_post, f_next: (Q, Nx, Ny, Nz)
        for i in range(self.lattice.Q):
            shift = (int(self.c[0, i]), int(self.c[1, i]), int(self.c[2, i]))
            
            f_next[i, ...] = self.xp.roll(f_post[i, ...], shift=shift, axis=(0, 1, 2))
