class StreamingPull:
    """Vectorized Pull-scheme Streaming Operator
    
    The pull scheme reads from neighboring cells:
        f_i(x, t+1) = f_i^post(x - c_i, t)
    
    This is implemented by precomputing source indices, enabling
    fully vectorized operations without Python loops.
    
    Attributes:
        xp: Array module (numpy or cupy)
        lattice: Lattice model containing velocity vectors
        shape: Domain shape (Nx, Ny, Nz) or (Nx, Ny)
    """
    def __init__(self, xp, lattice, shape):
        self.xp = xp
        self.lattice = lattice
        self.dim = lattice.dim
        self.shape = shape
        self.Q = lattice.Q

        self._precompute_indices()

    def _precompute_indices(self) -> None:
        """
        For pull scheme: src[i, x] = (x - c_i) mod N
        """
        xp = self.xp
        c = xp.asarray(self.lattice.c)

        Nx, Ny, Nz = self.shape
        ix = xp.arange(Nx, dtype=xp.int32)
        iy = xp.arange(Ny, dtype=xp.int32)
        iz = xp.arange(Nz, dtype=xp.int32)

        IX, IY, IZ = xp.meshgrid(ix, iy, iz, indexing='ij')

        # Compute source indices: (Q, Nx, Ny, Nz)
        # Pull scheme: source = current - velocity
        self.src_x = (IX[None, :, :, :] - c[0, :, None, None, None] + Nx) % Nx
        self.src_y = (IY[None, :, :, :] - c[1, :, None, None, None] + Ny) % Ny
        self.src_z = (IZ[None, :, :, :] - c[2, :, None, None, None] + Nz) % Nz

        self.src_x = self.src_x.astype(xp.int32)
        self.src_y = self.src_y.astype(xp.int32)
        self.src_z = self.src_z.astype(xp.int32)

        self.q_idx = xp.arange(self.Q, dtype=xp.int32)[:, None, None, None]
        self.q_idx = xp.broadcast_to(self.q_idx, (self.Q, Nx, Ny, Nz))
    
    def compute(self, f_post, f_next):
        """Perform streaming step (out-of-place)
        
        Args:
            f_post: Post-collision distribution
            f_next: Output buffer for streamed distribution
        """
        f_next[:] = f_post[self.q_idx, self.src_x, self.src_y, self.src_z]
    
    def stream_inplace(self, f):
        """Streaming with internal temporary buffer
        
        Args:
            f: Distribution function
            
        Returns:
            Streamed distribution (new array)
        """
        f_next = self.xp.empty_like(f)
        self.stream(f, f_next)
        return f_next

    # def compute(self, f_post, f_next):
    #     # f_post, f_next: (Q, Nx, Ny, Nz)
    #     for i in range(self.lattice.Q):
    #         shift = (int(self.c[0, i]), int(self.c[1, i]), int(self.c[2, i]))

    #         f_next[i, ...] = self.xp.roll(f_post[i, ...], shift=shift, axis=(0, 1, 2))
