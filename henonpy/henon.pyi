from numiphy.odesolvers import LowLevelODE, OdeResult

class HenonOde(LowLevelODE):

    def psolve(self, ics, t, dt, **kwargs)->OdeResult:...


ode: HenonOde