



def apply_wall_boundaries(xp, lattice, domain, boundaries):
    for boundary in boundaries.values():
        if boundary['type'] == 'wall':
            wall_location = boundary['location']
            method = boundary['method']
            q = boundary.get('q', 0.5)  # Bouzidi : q =0.5

            if method == 'bouzidi':
                for i in range(lattice.Q):
                    f[lattice.opp[i], wall_location] = f[i, wall_location] * q
    
    return f