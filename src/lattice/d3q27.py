
class D3Q27:
    def __init__(self, xp):
        self.xp = xp
        self.Q = 27
        self.dim = 3
        self.c = xp.array([
            [0,
            1, -1, 0, 0, 0, 0,
            1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0,
            1, -1,  1, -1,  1, -1,  1, -1],  # x
            [0,
             0,  0,  1, -1,  0,  0,
             1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1,
             1,  1, -1, -1,  1,  1, -1, -1],  # y
            [0,
             0,  0,  0,  0,  1, -1,
             0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1,
             1,  1,  1,  1, -1, -1, -1, -1]   # z
        ], dtype=xp.int32)

        self.w = xp.array([8/27] +
                           [2/27]*6 +
                           [1/54]*12 +
                           [1/216]*8, dtype=float)
        
        self.opp = self._compute_opposite_indices()
        
        self.cs2 = 1.0 / 3.0
    

    def _compute_opposite_indices(self):
        """벡터화된 방식으로 반대 방향 인덱스 계산"""
        opp = self.xp.zeros(self.Q, dtype=self.xp.int32)
        
        for i in range(self.Q):
            # 모든 방향과 -c[:, i]의 차이의 절댓값 합 계산
            diff = self.xp.abs(self.c + self.c[:, i:i+1]).sum(axis=0)
            # 차이가 0인 인덱스가 반대 방향
            opp[i] = self.xp.argmin(diff)
        
        return opp