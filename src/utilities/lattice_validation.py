"""
Lattice Model Validation Utility for LBM

This module validates isotropy conditions required for correct hydrodynamics
(Navier-Stokes recovery) in Lattice Boltzmann Methods.

Physical Requirements:
    For a lattice model to correctly recover the Navier-Stokes equations,
    the following isotropy conditions must be satisfied:
    
    1. Σ w_i = 1                              (normalization)
    2. Σ w_i c_iα = 0                         (zero momentum at rest)
    3. Σ w_i c_iα c_iβ = c_s² δ_αβ            (pressure tensor isotropy)
    4. Σ w_i c_iα c_iβ c_iγ = 0               (odd moments vanish)
    5. Σ w_i c_iα c_iβ c_iγ c_iδ = c_s⁴ Δ     (viscous stress isotropy)
    
    where:
        w_i     = weight for direction i  [dimensionless]
        c_i     = lattice velocity vector  [Δx/Δt]
        c_s²    = speed of sound squared = 1/3  [Δx²/Δt²]
        δ_αβ    = Kronecker delta
        Δ_αβγδ  = δ_αβδ_γδ + δ_αγδ_βδ + δ_αδδ_βγ  (isotropic 4th rank tensor)

References:
    - Kruger et al., "The Lattice Boltzmann Method", Springer 2017
    - Lallemand & Luo, Phys. Rev. E 61, 2000

Author: LBM Development Team
Date: 2026-01
"""

from typing import TYPE_CHECKING, Tuple, Dict, Optional
import numpy as np

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class LatticeValidator:
    """Validates isotropy conditions for LBM lattice models
    
    This class checks whether a given lattice model (weights and velocities)
    satisfies the mathematical constraints required to recover correct
    fluid dynamics from the Lattice Boltzmann equation.
    
    Attributes:
        xp: Array module (numpy or cupy)
        tol: Tolerance for floating-point comparisons
    """
    
    def __init__(self, xp: 'ModuleType' = None, tol: float = 1e-12) -> None:
        """Initialize validator
        
        Args:
            xp: Array module (defaults to numpy if None)
            tol: Tolerance for numerical comparisons  [dimensionless]
        """
        self.xp = xp if xp is not None else np
        self.tol = tol
    
    def validate_all(self, c: 'npt.NDArray', w: 'npt.NDArray', 
                     cs2: float, verbose: bool = True) -> Tuple[bool, Dict]:
        """Run all isotropy validation checks
        
        Validates the complete set of isotropy conditions required for
        correct Navier-Stokes recovery.
        
        Args:
            c: Lattice velocity vectors, shape (dim, Q)  [Δx/Δt]
            w: Lattice weights, shape (Q,)  [dimensionless]
            cs2: Speed of sound squared  [Δx²/Δt², typically 1/3]
            verbose: Print detailed results if True
            
        Returns:
            Tuple of:
                - bool: True if all conditions satisfied
                - dict: Detailed results for each check
        """
        xp = self.xp
        c = xp.asarray(c, dtype=xp.float64)
        w = xp.asarray(w, dtype=xp.float64)
        
        dim = c.shape[0]
        Q = c.shape[1]
        tol = self.tol
        
        results = {}
        all_passed = True
        
        if verbose:
            print("="*60)
            print(f" Lattice Validation: D{dim}Q{Q}")
            print("="*60)
        
        # =====================================================================
        # Check 1: Weight Normalization
        #   Σ w_i = 1
        # =====================================================================
        weight_sum = float(xp.sum(w))
        results['weight_sum'] = weight_sum
        passed = abs(weight_sum - 1.0) < tol
        all_passed &= passed
        
        if verbose:
            print(f"\n[1] Weight Normalization: Σw_i = 1")
            print(f"    Computed: {weight_sum:.15f}")
            print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        # =====================================================================
        # Check 2: Zero First Moment (rest state has zero momentum)
        #   Σ w_i c_iα = 0
        # =====================================================================
        first_moment = xp.einsum('i,ai->a', w, c)
        max_err = float(xp.max(xp.abs(first_moment)))
        results['first_moment_max_error'] = max_err
        passed = max_err < tol
        all_passed &= passed
        
        if verbose:
            print(f"\n[2] First Moment: Σw_i·c_iα = 0")
            print(f"    Max error: {max_err:.2e}")
            print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        # =====================================================================
        # Check 3: Second Moment Isotropy (pressure tensor)
        #   Σ w_i c_iα c_iβ = c_s² δ_αβ
        # =====================================================================
        second_moment = xp.einsum('i,ai,bi->ab', w, c, c)
        expected_second = cs2 * xp.eye(dim)
        second_error = xp.abs(second_moment - expected_second)
        max_err = float(xp.max(second_error))
        results['second_moment_max_error'] = max_err
        passed = max_err < tol
        all_passed &= passed
        
        if verbose:
            print(f"\n[3] Second Moment Isotropy: Σw_i·c_iα·c_iβ = c_s²·δ_αβ")
            print(f"    c_s² = {cs2:.10f}")
            print(f"    Max error: {max_err:.2e}")
            print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        # =====================================================================
        # Check 4: Third Moment Vanishes (odd moments are zero)
        #   Σ w_i c_iα c_iβ c_iγ = 0
        # =====================================================================
        third_moment = xp.einsum('i,ai,bi,ci->abc', w, c, c, c)
        max_err = float(xp.max(xp.abs(third_moment)))
        results['third_moment_max_error'] = max_err
        passed = max_err < tol
        all_passed &= passed
        
        if verbose:
            print(f"\n[4] Third Moment Vanishes: Σw_i·c_iα·c_iβ·c_iγ = 0")
            print(f"    Max error: {max_err:.2e}")
            print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        # =====================================================================
        # Check 5: Fourth Moment Isotropy (viscous stress)
        #   Σ w_i c_iα c_iβ c_iγ c_iδ = c_s⁴ · Δ_αβγδ
        #   where Δ_αβγδ = δ_αβδ_γδ + δ_αγδ_βδ + δ_αδδ_βγ
        # =====================================================================
        fourth_moment = xp.einsum('i,ai,bi,ci,di->abcd', w, c, c, c, c)
        
        # Construct expected fourth moment tensor
        cs4 = cs2 * cs2
        delta = xp.eye(dim)
        expected_fourth = cs4 * (
            xp.einsum('ab,cd->abcd', delta, delta) +  # δ_αβ δ_γδ
            xp.einsum('ac,bd->abcd', delta, delta) +  # δ_αγ δ_βδ
            xp.einsum('ad,bc->abcd', delta, delta)    # δ_αδ δ_βγ
        )
        
        fourth_error = xp.abs(fourth_moment - expected_fourth)
        max_err = float(xp.max(fourth_error))
        results['fourth_moment_max_error'] = max_err
        passed = max_err < tol
        all_passed &= passed
        
        if verbose:
            print(f"\n[5] Fourth Moment Isotropy: Σw_i·c_iα·c_iβ·c_iγ·c_iδ = c_s⁴·Δ_αβγδ")
            print(f"    c_s⁴ = {cs4:.10f}")
            print(f"    Max error: {max_err:.2e}")
            print(f"    Status: {'✓ PASS' if passed else '✗ FAIL'}")
        
        # =====================================================================
        # Summary
        # =====================================================================
        results['all_passed'] = all_passed
        
        if verbose:
            print("\n" + "="*60)
            if all_passed:
                print(" ✅ All isotropy conditions satisfied!")
                print("    This lattice model can correctly recover Navier-Stokes.")
            else:
                print(" ❌ Some isotropy conditions FAILED!")
                print("    Check the lattice velocity vectors and weights.")
            print("="*60)
        
        return all_passed, results
    
    def validate_opposite_indices(self, c: 'npt.NDArray', 
                                   opp: 'npt.NDArray',
                                   verbose: bool = True) -> Tuple[bool, Dict]:
        """Validate that opposite direction indices are correct
        
        For bounce-back boundary conditions, we need c[opp[i]] = -c[i].
        
        Args:
            c: Lattice velocities, shape (dim, Q)  [Δx/Δt]
            opp: Opposite direction indices, shape (Q,)
            verbose: Print results if True
            
        Returns:
            Tuple of (passed, details_dict)
        """
        xp = self.xp
        c = xp.asarray(c)
        opp = xp.asarray(opp)
        Q = c.shape[1]
        
        all_correct = True
        errors = []
        
        for i in range(Q):
            i_opp = int(opp[i])
            c_i = c[:, i]
            c_opp = c[:, i_opp]
            
            if not xp.allclose(c_i + c_opp, 0):
                all_correct = False
                errors.append((i, i_opp, c_i, c_opp))
        
        if verbose:
            print("\n[Opposite Direction Validation]")
            if all_correct:
                print(f"  ✓ All {Q} opposite indices are correct: c[opp[i]] = -c[i]")
            else:
                print(f"  ✗ Found {len(errors)} incorrect opposite mappings:")
                for i, i_opp, c_i, c_opp in errors[:5]:  # Show first 5
                    print(f"    i={i}: c[{i}]={c_i}, c[{i_opp}]={c_opp}, sum={c_i+c_opp}")
        
        return all_correct, {'errors': errors}


def validate_d2q9(xp: Optional['ModuleType'] = None, 
                  verbose: bool = True) -> Tuple[bool, Dict]:
    """Convenience function to validate D2Q9 lattice
    
    Args:
        xp: Array module (defaults to numpy)
        verbose: Print detailed results
        
    Returns:
        Tuple of (passed, details_dict)
    """
    if xp is None:
        xp = np
    
    from src.lattice.d2q9 import D2Q9
    
    lattice = D2Q9(xp)
    validator = LatticeValidator(xp)
    
    passed, results = validator.validate_all(
        lattice.c, lattice.w, lattice.cs2, verbose
    )
    
    opp_passed, opp_results = validator.validate_opposite_indices(
        lattice.c, lattice.opp, verbose
    )
    
    results['opposite_indices_valid'] = opp_passed
    results['opposite_indices_errors'] = opp_results.get('errors', [])
    
    return passed and opp_passed, results


def validate_d3q27(xp: Optional['ModuleType'] = None, 
                   verbose: bool = True) -> Tuple[bool, Dict]:
    """Convenience function to validate D3Q27 lattice
    
    Args:
        xp: Array module (defaults to numpy)
        verbose: Print detailed results
        
    Returns:
        Tuple of (passed, details_dict)
    """
    if xp is None:
        xp = np
    
    from src.lattice.d3q27 import D3Q27
    
    lattice = D3Q27(xp)
    validator = LatticeValidator(xp)
    
    passed, results = validator.validate_all(
        lattice.c, lattice.w, lattice.cs2, verbose
    )
    
    opp_passed, opp_results = validator.validate_opposite_indices(
        lattice.c, lattice.opp, verbose
    )
    
    results['opposite_indices_valid'] = opp_passed
    results['opposite_indices_errors'] = opp_results.get('errors', [])
    
    return passed and opp_passed, results


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate LBM lattice model")
    parser.add_argument('--lattice', type=str, default='D2Q9',
                        choices=['D2Q9', 'D3Q27', 'D3Q19'],
                        help='Lattice model to validate')
    args = parser.parse_args()
    
    print(f"\nValidating {args.lattice} lattice model...\n")
    
    if args.lattice == 'D2Q9':
        passed, _ = validate_d2q9(verbose=True)
    elif args.lattice == 'D3Q27':
        passed, _ = validate_d3q27(verbose=True)
    else:
        print(f"Validation for {args.lattice} not yet implemented.")
        passed = False
    
    exit(0 if passed else 1)