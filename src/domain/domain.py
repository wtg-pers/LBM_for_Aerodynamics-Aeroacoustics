class Domain:
    def __init__(self, lattice, xp, Nx, Ny, Nz):
        self.xp = xp
        self.lattice = lattice
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz

        self.dim = lattice.dim

        if self.dim == 2:
            self.f = xp.zeros((lattice.Q, Nx, Ny), dtype=xp.float64)
            self.f_eq = xp.zeros_like(self.f)
            self.rho = xp.ones((Nx, Ny), dtype=xp.float64)
            self.u = xp.zeros((2, Nx, Ny), dtype=xp.float64)

        elif self.dim == 3:
            if Nz is None:
                raise ValueError("Nz must be provided for 3D simulation")
            self.f = xp.zeros((lattice.Q, Nx, Ny, Nz), dtype=xp.float64)
            self.f_eq = xp.zeros_like(self.f)
            self.rho = xp.ones((Nx, Ny, Nz), dtype=xp.float64)
            self.u = xp.zeros((3, Nx, Ny, Nz), dtype=xp.float64)
        
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")



        