from ._integrate import *

class HenonOde(LowLevelODE):

    def __init__(self, q0: list, args: tuple, *, rtol=1e-6, atol=1e-12, min_step=0., event_tol=1e-12): ...

    def pintegrate(self, p: int)->OdeResult:...