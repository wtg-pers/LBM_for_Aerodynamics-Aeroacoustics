from typing import TYPE_CHECKING, Tuple, Optional

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt

class ConservationChecker:
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice') -> None:
        self.xp = xp
        self.lattice = lattice
        self.c = xp.asarray(lattice.c)
        
        self.initial_mass: Optional[float] = None
        self.initial_momentum: Optional['npt.NDArray'] = None
    
    def compute_mass(self, f: 'npt.NDArray') -> float:
        """Total mass from distribution"""
        return float(self.xp.sum(f))
    
    def compute_momentum(self, f: 'npt.NDArray') -> 'npt.NDArray':
        """Total momentum from distribution"""
        f_sum = self.xp.sum(f, axis=tuple(range(1, f.ndim)))
        return self.xp.dot(self.c, f_sum)
    
    def set_reference(self, f: 'npt.NDArray') -> None:
        """Set initial reference values"""
        self.initial_mass = self.compute_mass(f)
        self.initial_momentum = self.compute_momentum(f)
    
    def check_conservation(self, f: 'npt.NDArray', 
                          rtol: float = 1e-10,
                          verbose: bool = True) -> Tuple[bool, dict]:
        """Check conservation against reference"""
        if self.initial_mass is None:
            raise ValueError("Reference not set. Call set_reference() first.")
        
        current_mass = self.compute_mass(f)
        current_momentum = self.compute_momentum(f)
        
        mass_error = abs(current_mass - self.initial_mass) / (abs(self.initial_mass) + 1e-16)
        momentum_error = self.xp.linalg.norm(current_momentum - self.initial_momentum)
        momentum_error /= (self.xp.linalg.norm(self.initial_momentum) + 1e-16)
        
        mass_conserved = mass_error < rtol
        momentum_conserved = float(momentum_error) < rtol
        
        details = {
            'initial_mass': self.initial_mass,
            'current_mass': current_mass,
            'mass_error': mass_error,
            'mass_conserved': mass_conserved,
            'initial_momentum': self.initial_momentum,
            'current_momentum': current_momentum,
            'momentum_error': float(momentum_error),
            'momentum_conserved': momentum_conserved,
        }
        
        if verbose:
            print("="*60)
            print(" Conservation Check Results")
            print("="*60)
            print(f"  Mass:")
            print(f"    Initial:   {self.initial_mass:.15e}")
            print(f"    Current:   {current_mass:.15e}")
            print(f"    Rel Error: {mass_error:.6e}  {'✓' if mass_conserved else '✗'}")
            print(f"  Momentum:")
            init_mom_str = ', '.join([f'{m:.10e}' for m in self.initial_momentum])
            curr_mom_str = ', '.join([f'{m:.10e}' for m in current_momentum])
            print(f"    Initial:   [{init_mom_str}]")
            print(f"    Current:   [{curr_mom_str}]")
            print(f"    Rel Error: {float(momentum_error):.6e}  {'✓' if momentum_conserved else '✗'}")
            print("="*60)
        
        return (mass_conserved and momentum_conserved), details