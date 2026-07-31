"""
Command Line Argument Parser for LBM Solver

This module handles parsing of command-line arguments for the simulation.
Separates CLI logic from main simulation code for better modularity.

Author: LBM Development Team
Date: 2026-01
"""

import argparse
import os
from typing import Any, Dict, Optional, Sequence, Tuple

# MPI-only flags: dest -> (flag string, parser default). The entry dispatcher
# rejects any of these that deviate from its default when world size == 1,
# instead of silently ignoring them (no-silent rule).
MPI_ONLY_DESTS: Dict[str, Tuple[str, Any]] = {
    'steps': ('--steps', None),
    'devices': ('--devices', None),
    'axis': ('--axis', 'auto'),
    'ghost': ('--ghost', 3),
    'cuda_aware': ('--cuda-aware', None),
    'csv': ('--csv', None),
    'log_every': ('--log-every', None),
    'vtk_every': ('--vtk-every', None),
    'vtk_fields_last': ('--vtk-fields-last', 0),
    'ckpt_every': ('--ckpt-every', None),
    'dist_init': ('--dist-init', False),
    'verify': ('--verify', False),
    'strict_bit': ('--strict-bit', False),
    'profile': ('--profile', False),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments
    
    Argument Priority: CLI arguments > config file settings
    
    Returns:
        argparse.Namespace: Parsed arguments with following attributes:
            - config: Path to configuration file
            - restart: Path to specific checkpoint file
            - restart_latest: Boolean, restart from latest checkpoint
            - extend: Number of additional steps from restart point
            - max_steps: Override max_steps from config
            - output_dir: Override VTK output directory
            - checkpoint_dir: Override checkpoint directory
            - csv_dir: Override CSV output directory
            - clear: Boolean, clear previous results
            - no_vtk: Boolean, disable VTK output
            - no_force: Boolean, disable force calculation
    
    Examples:
        >>> args = parse_args()
        >>> print(args.config)
        './configs/input_config.py'
    """
    parser = argparse.ArgumentParser(
        description='LBM Solver for Aerodynamics/Aeroacoustics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Normal run with config settings
  python main.py --clear                      # Force clear previous results
  python main.py --restart-latest             # Continue from latest checkpoint
  python main.py --restart-latest --extend 5000
  python main.py --config ./configs/my_config.py
  python main.py --max-steps 50000            # Override max steps
        """
    )
    
    # ==========================================================================
    # Configuration
    # ==========================================================================
    parser.add_argument(
        '--config', type=str, default='./configs/input_config.py',
        help='Path to configuration file (default: ./configs/input_config.py)'
    )

    parser.add_argument(
        '--gpu', type=str, default=None,
        metavar='ID[,ID,...]',
        help='GPU device id(s). One id: this process\'s GPU (overrides the '
             'config file). Several comma-separated ids: MPI runs only — '
             'per node-local-rank mapping (rank r -> id[r %% len]), e.g. '
             'mpirun -n 2 ... --gpu 2,3. Unset: config device_id (single) '
             'or local_rank %% ndev (MPI).'
    )

    parser.add_argument(
        '--list-gpus', action='store_true',
        help='List available GPUs and exit'
    )
    
    # ==========================================================================
    # Restart Options
    # ==========================================================================
    restart_group = parser.add_argument_group('Restart Options')
    
    restart_group.add_argument(
        '--restart', type=str, default=None,
        metavar='CHECKPOINT_PATH',
        help='Path to checkpoint file for restart'
    )
    restart_group.add_argument(
        '--restart-latest', action='store_true',
        help='Restart from latest checkpoint in checkpoint directory'
    )
    restart_group.add_argument(
        '--extend', type=int, default=None,
        metavar='N',
        help='Extend simulation by N additional steps from restart point'
    )
    restart_group.add_argument(
        '--max-steps', type=int, default=None,
        metavar='N',
        help='Override max_steps from config (absolute end step)'
    )
    
    # ==========================================================================
    # Directory Overrides
    # ==========================================================================
    dir_group = parser.add_argument_group('Directory Overrides (CLI > config)')
    
    dir_group.add_argument(
        '--results-dir', type=str, default=None,
        metavar='PATH',
        help='Override the WHOLE result folder root: writes vtk/, csv/, and '
             'checkpoints/ under PATH (one flag renames the run folder). The '
             'per-subdir flags below take precedence if also given.'
    )
    dir_group.add_argument(
        '--vtk-dir', type=str, default=None,
        metavar='PATH',
        help='Override VTK output subdirectory (formerly --output-dir)'
    )
    dir_group.add_argument(
        '--checkpoint-dir', type=str, default=None,
        metavar='PATH',
        help='Override checkpoint subdirectory'
    )
    dir_group.add_argument(
        '--csv-dir', type=str, default=None,
        metavar='PATH',
        help='Override CSV output subdirectory'
    )
    
    # ==========================================================================
    # Output Control
    # ==========================================================================
    output_group = parser.add_argument_group('Output Control')
    
    output_group.add_argument(
        '--clear', action='store_true',
        help='Clear previous results before starting (overrides config)'
    )
    output_group.add_argument(
        '--no-vtk', action='store_true',
        help='Disable VTK output'
    )
    output_group.add_argument(
        '--no-force', action='store_true',
        help='Disable force calculation'
    )
    output_group.add_argument(
        '--verbose', action='store_true',
        help='Echo the captured setup log to the terminal as well'
    )

    # ==========================================================================
    # MPI (multi-rank only) — rejected by the dispatcher at world size == 1
    # ==========================================================================
    mpi_group = parser.add_argument_group(
        'MPI (multi-rank only; launch with mpirun -n N>1)')

    mpi_group.add_argument(
        '--mpi', action='store_true',
        help='Force the MPI path even if no MPI launcher env is detected '
             '(escape hatch for exotic launchers)'
    )
    mpi_group.add_argument(
        '--steps', type=int, default=None, metavar='N',
        help='[MPI] DEPRECATED alias of --max-steps N (same advance count; '
             'step labels are 0-based since the C8 unification)'
    )
    mpi_group.add_argument(
        '--axis', default='auto', choices=['auto', 'x', 'y', 'z'],
        help='[MPI] slab decomposition axis'
    )
    mpi_group.add_argument(
        '--devices', default=None, metavar='LIST',
        help="[MPI][DEPRECATED] alias of --gpu ID[,ID,...] — use --gpu; "
             "comma-separated GPU ids per node-local rank"
    )
    mpi_group.add_argument(
        '--ghost', type=int, default=3,
        help='[MPI] halo ghost width'
    )
    mpi_group.add_argument(
        '--cuda-aware', default=None, metavar='0|1',
        help='[MPI] pass CuPy buffers to MPI directly (default: env '
             'LBM_MPI_CUDA, 0 if unset)'
    )
    mpi_group.add_argument(
        '--csv', default=None, metavar='PATH',
        help='[MPI] dense timeseries CSV (rank 0)'
    )
    mpi_group.add_argument(
        '--log-every', type=int, default=None, metavar='N',
        help='[MPI] coarse steps between progress lines (default 8)'
    )
    mpi_group.add_argument(
        '--vtk-every', type=int, default=None, metavar='N',
        help='[MPI] coarse steps between assembled VTK writes (0=off)'
    )
    mpi_group.add_argument(
        '--vtk-fields-last', type=int, default=0, metavar='N',
        help='[MPI] write level-field VTK only for the last N vtk events'
    )
    mpi_group.add_argument(
        '--ckpt-every', type=int, default=None, metavar='N',
        help='[MPI] coarse steps between assembled checkpoints (0=off)'
    )
    mpi_group.add_argument(
        '--dist-init', action='store_true',
        help='[MPI] slab-scoped initialization: no full-size device fields '
             'per rank (uniform-IC cases)'
    )
    mpi_group.add_argument(
        '--verify', action='store_true',
        help='[MPI] compare assembled result vs a fresh 1-rank reference'
    )
    mpi_group.add_argument(
        '--strict-bit', action='store_true',
        help='[MPI] verify passes ONLY on bit-identity'
    )
    mpi_group.add_argument(
        '--profile', action='store_true',
        help='[MPI] per-section wall-time attribution'
    )

    args = parser.parse_args(argv)

    # ── --gpu normalization (unified single/MPI device flag) ─────────
    # args.gpu_ids: list[int] when --gpu was given, else None.
    # args.gpu:     int when exactly ONE id was given (single-process
    #               device override, historical type), else None — the
    #               MPI driver maps gpu_ids per node-local rank itself.
    args.gpu_ids = None
    if args.gpu is not None:
        try:
            ids = [int(t) for t in str(args.gpu).split(',') if t.strip()]
        except ValueError:
            parser.error(f"--gpu {args.gpu!r}: expected ID or ID,ID,... "
                         f"(integers)")
        if not ids:
            parser.error(f"--gpu {args.gpu!r}: no device id given")
        args.gpu_ids = ids
        args.gpu = ids[0] if len(ids) == 1 else None
    return args


def get_args_summary(args: argparse.Namespace) -> str:
    """Generate a summary string of parsed arguments
    
    Args:
        args: Parsed argument namespace
        
    Returns:
        Multi-line string summarizing non-default arguments
    """
    lines = ["Command Line Arguments:"]
    
    # Check each argument
    if args.config != './configs/input_config.py':
        lines.append(f"  Config: {args.config}")
    
    if args.restart_latest:
        lines.append("  Restart: from latest checkpoint")
    elif args.restart:
        lines.append(f"  Restart: {args.restart}")
    
    if args.extend:
        lines.append(f"  Extend: +{args.extend} steps")
    
    if args.max_steps:
        lines.append(f"  Max steps: {args.max_steps}")
    
    if args.vtk_dir:
        lines.append(f"  VTK dir: {args.vtk_dir}")
    
    if args.checkpoint_dir:
        lines.append(f"  Checkpoint dir: {args.checkpoint_dir}")
    
    if args.csv_dir:
        lines.append(f"  CSV dir: {args.csv_dir}")
    
    if args.clear:
        lines.append("  Clear previous: Yes")
    
    if args.no_vtk:
        lines.append("  VTK output: Disabled")
    
    if args.no_force:
        lines.append("  Force calculation: Disabled")
    
    if len(lines) == 1:
        lines.append("  (all defaults)")
    
    return "\n".join(lines)