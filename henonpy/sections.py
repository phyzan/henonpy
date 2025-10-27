from __future__ import annotations
from numiphy.toolkit import tools
from numiphy.toolkit import Template
from numiphy.toolkit import interpolate1D
from numiphy.toolkit.plotting import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure as Fig
from skimage.measure import find_contours
import shutil
from odepack import *

def _henon_heils_sys():
    x, y, px, py, t, wx, wy, eps, a, b, c = variables('x, y, px, py, t, wx, wy, eps, a, b, c')
    V = (wx**2*x**2 + wy**2*y**2)/2 + eps*(x*y**2 + a*x**3 + b*x**2*y +c*y**3)
    poinc_event = SymbolicEvent("PSoF", y, 1, event_tol=1e-20)
    return HamiltonianSystem2D(V, t, x, y, px, py, args=(eps, a, b, c, wx, wy), events=[poinc_event])

def henon_heiles_var_sys():
    x, y, px, py, delx, dely, delpx, delpy, t, wx, wy, eps, a, b, c = variables('x, y, px, py, delx, dely, delpx, delpy, t, wx, wy, eps, a, b, c')
    V = (wx**2*x**2 + wy**2*y**2)/2 + eps*(x*y**2 + a*x**3 + b*x**2*y +c*y**3)
    poinc_event = SymbolicEvent("PSoF", y, 1, event_tol=1e-20)
    return HamiltonianVariationalSystem2D(V, t, x, y, px, py, delx, dely, delpx, delpy, args=(eps, a, b, c, wx, wy), events=[poinc_event])

henon_heiles_system = _henon_heils_sys()

class Rat(float):

    def __new__(cls, m, n):
        return super().__new__(cls, m/n)
    
    def __init__(self, m, n):
        self.m = m
        self.n = n
    
    def __str__(self):
        return f'{self.m}/{self.n}'
    
    def __repr__(self):
        return str(self)
    
    def __getnewargs__(self) -> tuple[float]:
        return self.m, self.n


class HenonHeilesOrbit(LowLevelODE):

    E = 1

    eps: float
    alpha: float
    beta: float
    gamma: float
    omega_x: float
    omega_y: float
    is_flagged: bool
    is_active: bool

    def __init__(self, eps, alpha, beta, gamma, omega_x, omega_y, x0, px0, **kwargs):
        self.eps, self.alpha, self.beta, self.gamma, self.omega_x, self.omega_y, self.is_flagged, self.is_active = eps, alpha, beta, gamma, omega_x, omega_y, False, False

        py2 = 2*(self.E - self.V(x0, 0)) - px0**2
        if py2 < 0:
            raise ValueError('Kinetic energy is not positive')
        py0 = py2**0.5
        orbit = henon_heiles_system.get(0, [x0, 0, px0, py0], args=(eps, alpha, beta, gamma, omega_x, omega_y), **kwargs)
        super().__init__(orbit)

    @property
    def N(self):
        return self.t.shape[0]
    
    @property
    def color(self):
        if self.is_active:
            return 'forestgreen'
        elif self.is_flagged:
            return 'red'
        elif self.diverges:
            return 'k'
        else:
            return 'blue'
    
    @property
    def x(self):
        return self.q[:, :2].transpose()

    @property
    def p(self):
        return self.q[:, 2:].transpose()
    
    @property
    def xcoords(self):#because it is symmetric
        if self.beta == 0 and self.gamma == 0:
            return np.concatenate((self.x[0], self.x[0]), axis=0)
        else:
            return self.x[0]
        
    @property
    def ycoords(self):#because it is symmetric
        if self.beta == 0 and self.gamma == 0:
            return np.concatenate((self.p[0], -self.p[0]), axis=0)
        else:
            return self.p[0]

    def V(self, x, y):
        return 1/2*(self.omega_x**2*x**2 + self.omega_y**2*y**2) + self.eps*(x*y**2 + self.alpha*x**3 + self.beta*x**2*y + self.gamma*y**3)

    @property
    def title(self):
        numbers = (self.eps, self.alpha, self.x[0, 0], self.x[1, 0], self.p[0, 0], self.p[1, 0])
        strs = tuple([tools.format_number(round(n, 10))[1:-1] for n in numbers])
        return '$\\epsilon = %s, \\alpha = %s$\n$x_0 = %s, y_0 = %s, \\dot{x}_0 = %s, \\dot{y}_0 = %s$'%strs
    
    @staticmethod
    def pintegrate_all(orbs: list[HenonHeilesOrbit], N):
        _orbs = [orb for orb in orbs if not orb.is_dead]
        integrate_all(_orbs, 1e20, 0, [EventOpt("PSoF", max_events=N, terminate=True)], display_progress=False)


class HenonHeiles(Template):

    eps: float
    a: float
    b: float
    c: float
    w1: float
    w2: float
    E: float

    orbit_list: list[HenonHeilesOrbit]
    _temp: PointCollection
    _artists: list[PointCollection]

    def __init__(self, *, eps, a, b=0, c=0, w1=1, w2=1, E=1):
        Template.__init__(self, eps=eps, a=a, b=b, c=c, w1=w1, w2=w2, E=E, orbit_list=[], _temp=ScatterPlot(x=[], y=[], c='forestgreen', s=1), _artists=[])

    def V(self, x, y): 
        return 1/2*(self.w1**2*x**2 + self.w2**2*y**2) + self.eps*(x*y**2 + self.a*x**3 + self.b*x**2*y + self.c*y**3)

    def newcopy(self):
        hh = self.clone()
        hh._set(orbit_list=[], _temp=[], _artists=[])
        return hh
    
    def new_orbit(self, x, px, **kwargs):
        return HenonHeilesOrbit(*self.coefs, x, px, **kwargs)

    def init(self, x, px, **kwargs):
        orb = self.new_orbit(x, px, **kwargs)
        self.orbit_list.append(orb)
        return orb

    def init_all(self, q0, **kwargs):
        for qi in q0:
            self.init(*qi, **kwargs)

    def deactivate_all(self):
        for orb in self.orbit_list:
            orb.is_active = False

    def px_limits(self, x, y=0, py=1e-6):
        p2 = 2*(self.E-self.V(x, y)-py**2)
        if p2 < 0:
            raise ValueError(f'The point with coordinates x={x}, y={y}, p_y = {py} is outside the region of positive kinetic energy')
        p = p2**0.5
        return -p, p

    def perform_all(self, N):
        HenonHeilesOrbit.pintegrate_all(self.orbit_list, N)

    def periodic_orbit_near(self, x0, px0, n, derr=1e-8, **odekw)->tuple[float, float]:

        def dist(q):
            x, px = q
            orb = self.new_orbit(x, px, **odekw)
            orb.integrate(1e10, max_frames=0, event_options=[opt])
            return np.array([orb.x[0][-1]-x, orb.p[0][-1]-px])
        
        opt = EventOpt('PSoF', max_events=n, terminate=True)

        return fsolve(dist, [x0, px0], xtol=derr)

    def rotation_number(self, xobs, pxobs, x0, px0, N=300, ode={}):
        orb = self.new_orbit(x0, px0, **ode)
        orb.pintegrate(N)
        x = [xi-xobs for xi in orb.x]
        px = [pxi-pxobs for pxi in orb.p[0]]
        z = [xi+1j*pxi for xi, pxi in zip(x, px)]
        n = len(z)-1
        Deltaphi = [np.angle(z[i+1]/z[i]) for i in range(n)]
        return sum(Deltaphi)/(2*np.pi*n)

    def full_rot_num(self, xobs, pxobs, split=200, N=300, axis='x', ode={}):
        if axis == 'x':
            a, b = self.region()
        elif axis == 'px':
            a, b = self.px_limits(x=xobs, y=0)
        a = a+1e-6
        b = b-1e-6

        arr = np.linspace(a, b, split)
        if axis == 'x':
            q = [[xi, pxobs] for xi in arr]
        elif axis == 'px':
            q = [[xobs, pxi] for pxi in arr]
        ps = self.newcopy()
        ps.init_all(q)
        ps.perform_all(N, **ode)
        rot = []
        arr_final = []
        for j, orb in enumerate(ps.orbit_list):
            x = [xi-xobs for xi in orb.x[0]]
            px = [pxi-pxobs for pxi in orb.p[0]]
            if axis == 'x':
                z = [xi+1j*pxi for xi, pxi in zip(x, px)]
            elif axis == 'px':
                z = [pxi-xi*1j for xi, pxi in zip(x, px)]
            Deltaphi = [np.angle(z[i+1]/z[i]) for i in range(N-1) if z[i] != 0]
            if len(Deltaphi) < N-1:
                continue
            arr_final.append(arr[j])
            rot.append(sum(Deltaphi)/(2*np.pi*(N-1)))
        arr_final = np.array(arr_final)
        return arr, interpolate1D(arr_final, rot, arr)

    def nearest_orbit(self, x, px):
        _min = np.inf
        for orb in self.orbit_list:
            xarr, yarr = np.array([orb.x[0], orb.p[0]])
            d = np.min((xarr-x)**2 + (yarr-px)**2)
            if d < _min:
                _min = d
                _orb = orb
        return _orb

    def _draw(self, fig: Fig, ax: Axes, current_lims=True):
        if current_lims:
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
        else:
            xlims = (None, None)
            ylims = (None, None)
        ax.clear()
        self.figure.draw(ax)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        fig.canvas.draw_idle()

    def handle(self, event: MouseEvent, N: int, fig:Fig, ax:Axes, **odeargs):
        x, px = event.xdata, event.ydata
        if event.button == 1:
            if event.key is None:
                self._temp.add(x, px)
            elif event.key == 'delete':
                for i in range(self.Norbs-1, -1, -1):
                    if self.orbit_list[i].is_active:
                        self.orbit_list.pop(i)
                self._temp.clear()
            elif event.key == 'control':
                self.nearest_orbit(x, px).is_active = True
            elif event.key == 'shift':
                self._temp.clear()
                self.deactivate_all()
        elif event.button == 3 and event.key is None:
            if not self._temp.isempty():
                self.init_all(self._temp.all_coords(), **odeargs)
                Nnew = len(self._temp.all_coords())
                for orb in self.orbit_list[self.Norbs-Nnew:]:
                    orb.is_active = True
                self._temp.clear()

            active_orbs = [orb for orb in self.orbit_list if orb.is_active]
            if active_orbs:
                HenonHeilesOrbit.pintegrate_all(active_orbs, N)

        self._draw(fig, ax)

    def plot(self, interact = False, N=200, **odeargs):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal', adjustable='box')

        self._draw(fig, ax, current_lims=False)
        if interact:
            fig.canvas.mpl_connect('button_press_event', lambda event: self.handle(event, N, fig, ax, **odeargs))
            
        return fig, ax

    def remove_divergent(self):
        for i in range(self.Norbs-1, -1, -1):
            if self.orbit_list[i].diverges:
                self.orbit_list.pop(i)

    def intervals(self, px=0.):
        '''
        we demand that T = E - V(x, y=y) >= 0
        '''
        y=0
        def f(x):
            return self.E - self.V(x, y) - px**2/2
        kappa = self.a*self.eps
        if kappa == 0:
            xc = -self.eps*y**2/self.w1**2
            if f(xc) <= 0:
                return ()
            else:
                return sorted(np.roots([-self.w1**2/2, -self.eps*y**2, self.E-1/2*self.w2**2*y**2-1/2*px**2]).real),

        roots = roots_of_poly3(-kappa, -self.w1**2/2, -self.eps*y**2, self.E-1/2*self.w2**2*y**2-1/2*px**2)

        x = [-np.inf, *roots, np.inf]
        with np.errstate(all='ignore'):
            intervals = ()
            for i in range(len(x)-1):
                x1, x2 = x[i], x[i+1]
                if x1 == -np.inf:
                    xm = x2-1
                elif x2 == np.inf:
                    xm = x1 + 1
                else:
                    xm = (x1+x2)/2
                if f(xm) >= 0:
                    intervals += [x1, x2],
        return intervals

    def region(self):
        intervals = self.intervals()
        if np.all([np.isinf(inter).any() for inter in intervals]):
            raise ValueError(f'Infinite interval for eps={self.eps}, alpha={self.a}: {intervals}')
        for inter in intervals:
            if not np.isinf(inter).any():
                xmin, xmax = inter
                break
        return xmin, xmax

    def shell(self, xlims, ylims, n=1000):
        
        def f(x, px):
            return 1/2*px**2 + self.V(x, 0) - self.E
        
        data = implicit_plot_data(f, xlims, ylims, n)
        for (x, px) in data:
            self._artists.append(LinePlot(x=x, y=px, c='red', zorder=4, linewidth=2))
        

    def zero_vel_curve(self, xlims, ylims, n=1000, linewidth=2, **kwargs):
        def f(x, y):
            return self.V(x, y) - self.E
        
        fig = SquareFigure('czv_'+self.name, title=self.title, xlabel='$x$', ylabel='$y$', yrot=0, aspect='equal')

        data = implicit_plot_data(f, xlims, ylims, n)
        for (x, y) in data:
            fig.add(LinePlot(x=x, y=y, c='red', zorder=4, linewidth=linewidth, **kwargs))
        
        return fig
    
    def get_zero_vel_curve_roots(self):
        try:
            x1, x2, x3 = roots_of_poly3(-2*self.eps*self.a, -self.w1**2, 0, 2*self.E)
        except ValueError:
            return roots_of_poly3(-2*self.eps*self.a, -self.w1**2, 0, 2*self.E)
        if self.a >= 0:
            return x3, x2, x1
        else:
            return x1, x2, x3
        
    def plot_rot_num(self, xobs, pxobs, multiplicity, split=200, N=300, axis='x', save='', ode={}, **kwargs):
        x, rot = self.full_rot_num(xobs, pxobs, split, N, axis, ode)
        fig, ax = plt.subplots()
        ax.plot(x, rot, **kwargs)
        ax.set_xlabel(axis)
        ax.set_ylabel('Rotation Number')
        ax.set_title(self.title+f'\nPeriodic Orbit at x = {tools.format_number(xobs)}, $p_x$ = {tools.format_number(pxobs)} with n = {multiplicity}')
        ax.grid(True)
        
        if save != '':
            p1 = os.path.join(self.project_folder, f'{save}.png')
            p2 = os.path.join(self.project_folder, f'{save}.pdf')
            fig.savefig(p1, bbox_inches='tight', dpi=300)
            fig.savefig(p2, bbox_inches='tight')
        return fig, ax

    def get_limiting_px(self, pxmin, pxmax, x=0., bisect_xtol=1e-6, N=100, **odekw):

        def f(px):
            orb = self.new_orbit(x, px, **odekw)
            orb.pintegrate(N)
            return orb.diverges - 0.5
        
        return tools.bisect(f, pxmin, pxmax, tol=bisect_xtol)[1]
    
    def implicit(self, Phi, xlims, ylims, n=400, linewidth=1, c='brown', **kwargs):

        '''
        Phi(x, y, px, py, eps, a, b, c, w1, w2)
        '''

        def f(x, px):
            py = (2*(self.E - self.V(x, 0)) - px**2)**0.5
            return Phi(x, 0, px, py, *self.coefs) - Phi(orb.x[0, 0], 0, orb.p[0, 0], orb.p[1, 0], *self.coefs)
        
        fig = self.figure
        fig._artists.clear()
        for orb in self.orbit_list:
            data = implicit_plot_data(f, xlims, ylims, n)
            for (x, px) in data:
                fig.add(LinePlot(x=x, y=px, linewidth=linewidth, c=c, **kwargs))
                self._artists.append(fig.artists[-1])
        return fig
    
    def save_for_paper(self, proj: Project, name: str):
        save_for_paper(proj, self.figure, name)

    @property
    def Norbs(self):
        return len(self.orbit_list)

    @property
    def figure(self):
        fig = SquareFigure(self.name, title=self.title, xlabel='$x$', ylabel='$\\dot{x}$', yrot=0, aspect='equal')
        for art in self.artists:
            fig.add(art)
        return fig

    
    @property
    def coefs(self):
        return self.eps, self.a, self.b, self.c, self.w1, self.w2
    
    @property
    def artists(self):
        return [self._temp] + [ScatterPlot(x=orb.xcoords, y=orb.ycoords, c=orb.color, linewidth=0, s=1) for orb in self.orbit_list] + self._artists

    @property
    def name(self):
        A = round(self.w1**2, 10)
        B = round(self.w2**2, 10)
        return f'{self.eps}'.replace('.', '').replace('/', '_') + '__' + f'{self.a}'.replace('.', '').replace('/', '_') + '__' + f'{self.b}'.replace('.', '').replace('/', '_') + '__' + f'{self.c}'.replace('.', '').replace('/', '_') + '__' + f'{A}'.replace('.', '').replace('/', '_')+'__' + f'{B}'.replace('.', '').replace('/', '_')

    @property
    def title(self):
        return '$\\epsilon={0}, \\alpha={1}, \\beta={2}, \\gamma={3}$\n$\\omega_1={4}, \\omega_2={5}$'.format(*self.coefs)



class IntegrableHenonHeiles(HenonHeiles):

    _a: str
    Alpha: float

    def __init__(self, eps, w1=None, w2=None):
        if self._a != '16/3':
            w1 = 1 if w1 is None else w1
            w2 = 1 if w2 is None else w2
        else:
            if w1 is None and w2 is None:
                w2 = 1
                w1 = 4
            elif w1 is None:
                w1 = 4*w2
            elif w2 is None:
                w2 = w1/4
            else:
                assert w1 == 4*w2
        super().__init__(eps=eps, a=self.Alpha, b=0, c=0, w1=w1, w2=w2)

    @property
    def _a(self):
        return self.__class__._a


    def Q(self, x, y, px, py)->float|np.ndarray:...

    def K(self, x, px)->float|np.ndarray:...

    def Kplot(self, xmin, xmax, px=0., n=1000, **kwargs):
        fig = Figure('K_'+self.name, title=self.title, xlabel='x', ylabel='K', yrot=0, square=True)
        N = len(self.intervals(px=px))
        label = kwargs.pop('label', None)
        for i, inter in enumerate(self.intervals(px=px)):
            x1, x2 = inter
            if i == 0:
                if np.isinf(x1):
                    x1 = xmin
                else:
                    x1 = max(x1, xmin)
            if i == N-1:
                if np.isinf(x2):
                    x2 = xmax
                else:
                    x2 = min(x2, xmax)
            x = np.linspace(x1+1e-5, x2-1e-5, n)
            if i == 0 and label is not None:
                fig.add(LinePlot(x=x, y=self.K(x, px), **kwargs, label=label))
            else:
                fig.add(LinePlot(x=x, y=self.K(x, px), **kwargs))

        return fig
    
    def add_invariant(self, k, n=1000, xlims=(-5, 5), pxlims=(-5, 5), c='k', **kwargs):
        def K(x, px):
            return self.K(x, px) - k
        
        data = implicit_plot_data(K, xlims, pxlims, resolution=n)
        label = kwargs.pop('label', '')
        for i, (x, px) in enumerate(data):
            if i == 0 and label != '':
                self._artists.append(LinePlot(x=x, y=px, linewidth=2, c=c, **kwargs, label=label))
            else:
                self._artists.append(LinePlot(x=x, y=px, linewidth=2, c=c, **kwargs))

    def get_limiting_x(self, xmin, xmax=None, n=200):
        '''
        The K invariant such that there is no region with no escaping orbits
        '''

        def f(x):
            orb = ps.new_orbit(x, 0)
            orb.pintegrate(5)
            if orb.diverges:
                return 1
            else:
                return -1
            
        if xmax is None:
            xmax = self.intervals()[-1][-1]-1e-6

        x = np.linspace(xmin, xmax, n)
        q = [[xi, 0] for xi in x]
        ps = self.newcopy()
        ps.init_all(q)
        ps.perform_all(5)
        for i in range(n-1):
            for j in range(i+1, n):
                if (ps.orbit_list[i].diverges-0.5) * (ps.orbit_list[j].diverges-0.5) < 0:
                    return tools.bisect(f, x[i], x[j], tol=1e-5)[2]
        raise ValueError('Bisection limits invalid')


class IntHenon2(IntegrableHenonHeiles):

    Alpha = 2

    def Q(self, x, y, px, py):
        A, B = self.w1**2, self.w2**2
        Lxy = x*py - y*px
        return -4*Lxy*py + (4*B*x+self.eps*y**2+4*self.eps*x**2)*y**2+(4*B-A)/self.eps*(py**2 + B*y**2)

    def K(self, x, px):
        py2 = 2*(self.E-self.V(x, 0)) - px**2
        A, B = self.w1**2, self.w2**2
        return ((4*B-A)/self.eps-4*x)*py2
    
    def dKdx_roots(self)->list[float]:
        A, B = self.w1**2, self.w2**2
        l = (4*B-A)/self.eps
        return roots_of_poly3(32*self.eps*self.a, 3*(4*A-2*l*self.eps*self.a), -2*l*A, -8*self.E)
    
    def per_orbits(self)->list[float]:
        res = []
        for x in self.dKdx_roots():
            for x1, x2 in self.intervals():
                if x1 < x < x2:
                    orb = self.new_orbit(x, 0)
                    orb.pintegrate(10)
                    if not orb.diverges:
                        res.append(float(x))
        return res

    def perorb_range(self)->list[tuple[float, float, float]]:
        assert len(self.intervals()) == 1
        res = []
        for x in self.per_orbits():
            x1 = self.get_limiting_x(-5, x, n=2)
            x2 = self.get_limiting_x(x, n=2)
            res.append((x1, x, x2))
        return res


class IntHenon1_3(IntegrableHenonHeiles):

    Alpha = Rat(1, 3)

    def Q(self, x, y, px, py):
        return px*py + x*y + self.eps*(y*x**2 + y**3/3)

    def K(self, x, px):
        py2 = 2*(self.E-self.V(x, 0)) - px**2
        return (px**2*py2)**0.5


class IntHenon16_3(IntegrableHenonHeiles):

    Alpha = Rat(16, 3)

    def Q(self, x, y, px, py):
        B = self.w2**2 #A = 16*B already satisfied
        return py**4 + 2*(B+2*self.eps*x)*y**2*py**2-4/3*self.eps*y**3*px*py+B**2*y**4-4/3*self.eps*(B+self.eps*x)*y**4*x-2/9*self.eps**2*y**6

    def K(self, x, px):
        py2 = 2*(self.E-self.V(x, 0)) - px**2
        return py2**2



def recreate_folder(name, directory):
    """Check if the folder exists in the current working directory.
    If it exists, delete it and create an new empty one.
    If it doesn't exist, create the folder."""
    # Get the current working directory
    
    # Construct the full path for the folder
    folder_path = os.path.join(directory, name)
    
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Remove the existing folder and its contents
        shutil.rmtree(folder_path)
    
    # Create an empty folder
    os.makedirs(folder_path)
    return folder_path

def nroots_of_poly3(a, b, c, d):

    def f(x):
        return a*x**3 + b*x**2 + c*x + d

    _a, _b, _c = 3*a, 2*b, c

    D = _b**2 - 4*_a*_c
    if D > 0:
        x1 = (-_b - D**0.5)/(2*_a)
        x2 = (-_b + D**0.5)/(2*_a)
        if f(x1)*f(x2) < 0:
            return 3
    return 1
    
def roots_of_poly3(a, b, c, d):
    n = nroots_of_poly3(a, b, c, d)
    r = np.roots([a, b, c, d])
    p = abs(r.imag).argsort()
    r = r[p].real
    if n == 1:
        return [r[0]]
    else:
        return sorted(r)


def implicit_plot_data(f, xlim: tuple, ylim: tuple, resolution=400):
    """
    Computes the implicit function f(x, y) = 0 and extracts contour line data.

    Parameters:
    f : function
        A function of two variables, f(x, y).
    xlim : tuple
        The range of x-values (xmin, xmax).
    ylim : tuple
        The range of y-values (ymin, ymax).
    resolution : int, optional
        The number of points per axis for evaluation.

    Returns:
    list of (x, y) arrays representing separate contour line segments.
    """
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    with np.errstate(all='ignore'):
        Z = f(X, Y)

    # Find contours using skimage
    contours = find_contours(Z, level=0)

    # Transform contour coordinates from pixel indices to (x, y) values
    contour_lines = []
    for contour in contours:
        x_contour = x[0] + contour[:, 1] * (x[-1] - x[0]) / (Z.shape[1] - 1)
        y_contour = y[0] + contour[:, 0] * (y[-1] - y[0]) / (Z.shape[0] - 1)
        contour_lines.append((x_contour, y_contour))

    return contour_lines

def save_for_paper(proj: Project, fig: Figure, name: str):
    fig = fig.copy()
    fig.name = name+'_titled'
    proj.savefig(fig)
    fig2 = fig.copy()
    fig2.title = ''
    fig2.name = fig.name[:-7]
    proj.savefig(fig2)
    