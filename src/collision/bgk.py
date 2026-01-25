class BGK:
    def __init__(self, xp):
        self.xp = xp
    
    def collide(self, f, f_eq, tau):
        omega = 1 / tau
        f_neq = f - f_eq
        
        return f - omega * f_neq
