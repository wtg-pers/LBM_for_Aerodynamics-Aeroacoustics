"""
Command Line Argument Parser for LBM Solver

This module handles parsing of command-line arguments for the simulation.
Separates CLI logic from main simulation code for better modularity.

Author: LBM Development Team
Date: 2026-01
"""

import argparse
from typing import Any


def parse_args() -> argparse.Namespace:
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
        '--gpu', type=int, default=None,
        metavar='ID',
        help='GPU device ID to use (0, 1, 2, ...). Overrides config file.'
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
        '--output-dir', type=str, default=None,
        metavar='PATH',
        help='Override VTK output directory'
    )
    dir_group.add_argument(
        '--checkpoint-dir', type=str, default=None,
        metavar='PATH',
        help='Override checkpoint directory'
    )
    dir_group.add_argument(
        '--csv-dir', type=str, default=None,
        metavar='PATH',
        help='Override CSV output directory'
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
    
    return parser.parse_args()


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
    
    if args.output_dir:
        lines.append(f"  Output dir: {args.output_dir}")
    
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