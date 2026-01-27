from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Tuple, List, Optional, Dict, Any
from enum import Enum

if TYPE_CHECKING:
    from types import ModuleType
    import numpy.typing as npt


class BoundaryLocation(Enum):
    """Boundary face locations for 3D domain"""
    WEST = 'west'
    EAST = 'east'
    SOUTH = 'south'
    NORTH = 'north'
    BOTTOM = 'bottom'
    TOP = 'top'

class BoundaryCondition(ABC):
    def __init__(self, xp: 'ModuleType', lattice: 'Lattice',
                 location: BoundaryLocation) -> None:
        self.xp = xp
        self.lattice = lattice
        self.location = location
        self.c = xp.asarray(lattice.c)
        self.w = xp.asarray(lattice.w)
        self.Q = lattice.Q
        self.cs2 = lattice.cs2

        self.incoming_indices = self._get_incoming_indices()
    
    def _get_incoming_indices(self) -> 'npt.NDArray':
        xp = self.xp
        c = self.c

        if self.location == BoundaryLocation.WEST:
            mask = c[0, :] > 0
        elif self.location == BoundaryLocation.EAST:
            mask = c[0, :] < 0
        elif self.location == BoundaryLocation.SOUTH:
            mask = c[1, :] > 0
        elif self.location == BoundaryLocation.NORTH:
            mask = c[1, :] < 0
        elif self.location == BoundaryLocation.BOTTOM:
            mask = c[2, :] > 0
        elif self.location == BoundaryLocation.TOP:
            mask = c[2, :] < 0
        else:
            raise ValueError(f"Unknown location: {self.location}")
        
        return xp.where(mask)[0]
    
    @abstractmethod
    def apply(self, f: 'npt.NDArray', **kwargs) -> None:
        pass

    def get_boundary_slice(self, shape: Tuple[int, ...]) -> Tuple[slice, ...]:
        Nx, Ny, Nz = shape

        if self.location == BoundaryLocation.WEST:
            return (slice(None), 0, slice(None), slice(None))
        elif self.location == BoundaryLocation.EAST:
            return (slice(None), Nx-1, slice(None), slice(None))
        elif self.location == BoundaryLocation.SOUTH:
            return (slice(None), slice(None), 0, slice(None))
        elif self.location == BoundaryLocation.NORTH:
            return (slice(None), slice(None), Ny-1, slice(None))
        elif self.location == BoundaryLocation.BOTTOM:
            return (slice(None), slice(None), slice(None), 0)
        elif self.location == BoundaryLocation.TOP:
            return (slice(None), slice(None), slice(None), Nz-1)
    

class BoundaryManager:
    def __init__(self) -> None:
        self.boundaries: List[BoundaryCondition] = []
    
    def add(self, bc: BoundaryCondition) -> None:
        self.boundaries.append(bc)
    
    def apply_all(self, f: 'npt.NDArray', **kwargs) -> None:
        for bc in self.boundaries:
            bc.apply(f, **kwargs)
            
