from __future__ import annotations
from numiphy import odesolvers as ods
from numiphy.odesolvers import bisect, Bisect, bisectright
from numiphy.toolkit import tools
from numiphy.toolkit import interpolate1D
from numiphy.toolkit.plotting import *
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import os
from matplotlib.axes import Axes
from typing import Dict, Any, Type, Literal, Callable
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure as Fig
import shutil
from .henon import ode

_hhode = ode()


class HenonHeilesOrbit(ods.HamiltonianOrbit):

    E = 1

    eps: float
    alpha: float
    beta: float
    gamma: float
    omega_x: float
    omega_y: float
    is_flagged: bool
    is_active: bool

    def __init__(self, eps, alpha, beta, gamma, omega_x, omega_y, x0, px0):
        data = np.empty((0, 5), dtype=np.float64)
        ods.Base.__init__(self, ode=_hhode, data=data, diverges=False, is_stiff=True, eps=eps, alpha=alpha, beta=beta, gamma=gamma, omega_x=omega_x, omega_y=omega_y, is_flagged=False, is_active=False)

        py2 = 2*(self.E - self.V(x0, 0)) - px0**2
        if py2 < 0:
            raise ValueError('Kinetic energy is not positive')
        py0 = py2**0.5
        self.set_ics(0., [float(x0), 0., float(px0), py0])

    @property
    def px(self):
        return self.p[0]
    
    @property
    def py(self):
        return self.p[1]
    
    @property
    def x0(self):
        return self.x[0, 0]
    
    @property
    def y0(self):
        return self.x[1, 0]

    @property
    def px0(self):
        return self.px[0]
    
    @property
    def py0(self):
        return self.py[0]

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
    def xcoords(self):#because it is symmetric
        return self.x[0]
        return np.concatenate((self.x[0], self.x[0]), axis=0)
    
    @property
    def ycoords(self):#because it is symmetric
        return self.px
        return np.concatenate((self.px, -self.px), axis=0)
    
    def integrate(self, Delta_t, dt, **kwargs):
        return super().integrate(Delta_t, dt, args=(self.eps, float(self.alpha), float(self.beta), float(self.gamma), self.omega_x, self.omega_y), **kwargs)

    def pintegrate(self, N, dt=1e-2, err=1e-8):
        return self.integrate(1e10, dt, func = "psolve", err=err, max_frames=N+1)
    
    def reset(self):
        self._set(is_flagged=False, is_active=False)
        super().reset()

    def V(self, x, y):
        return 1/2*(self.omega_x**2*x**2 + self.omega_y**2*y**2) + self.eps*(x*y**2 + self.alpha*x**3 + self.beta*x**2*y + self.gamma*y**3)

    @property
    def title(self):
        numbers = (self.eps, self.alpha, self.x0, self.y0, self.px0, self.py0)
        strs = tuple([tools.format_number(round(n, 10))[1:-1] for n in numbers])
        return '$\\epsilon = %s, \\alpha = %s$\n$x_0 = %s, y_0 = %s, \\dot{x}_0 = %s, \\dot{y}_0 = %s$'%strs

    def copy(self):
        orb = HenonHeilesOrbit(self.eps, self.alpha, self.beta, self.gamma, self.omega_x, self.omega_y, self.x0, self.px0)
        orb._copy_data_from(self)
        return orb


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
    

class PSFig(SquareFigure):

    def __init__(self, ps: PoincareSection, **kwargs):
        self.ps = ps
        args = ps.proj.figparams
        args.update(title=ps.title, xlabel='x', ylabel='$\\dot{x}$', yrot=0)
        args.update(**kwargs)
        super().__init__(ps.name, **args)

    @property
    def artists(self):
        return self._artists + [self.ps.proj.scatter(x=orb.xcoords, y=orb.ycoords, c=orb.color) for orb in self.ps]

    def copy(self):
        fig = PSFig(self.ps)
        fig.set(**self.parameters)
        fig._artists = [artist.copy() for artist in self._artists]
        return fig


class PoincareSection:


    orbit: Dict[tuple, HenonHeilesOrbit]

    _ax: Axes
    _fig: Fig
    figure: PSFig

    _args: tuple[float]#extra parameters to pass when calling HenonHeilesOrbit, and in self.__class__
    fig: Figure
    E = 1

    def __init__(self, project_folder=os.getcwd()):
        self.orbit = {}
        self.proj = Project(project_folder)
        self.figure = PSFig(self)

    @property
    def name(self)->str:...

    @property
    def title(self):...

    @property
    def temp(self):
        return self.figure._artists[0]

    # def lims(self)->tuple[tuple[float, float], tuple[float, float]]:...

    def newcopy(self, load=False):
        return self.__class__(*self._args, load=load)

    def new_orbit(self, x, px):
        return HenonHeilesOrbit(*self._args, x, px)

    def init(self, x, px):
        orb = self.new_orbit(x, px)
        self.orbit[(x, px)] = orb
        return orb

    def remove(self, x0, px0):
        self.orbit.pop((x0, px0))

    def activate(self, x0, px0):
        self.orbit[(x0, px0)]._set(is_active=True)
    
    def deactivate(self, x0, px0):
        self.orbit[(x0, px0)]._set(is_active=False)

    def deactivate_all(self):
        for orb in self:
            orb._set(is_active=False)

    def __iter__(self):
        for qi in self.orbit:
            yield self.orbit[qi]

    def px_limits(self, x, y=0, py=1e-6):
        p2 = 2*(self.E-self.V(x, y)-py**2)
        assert p2 >= 0
        p = p2**0.5
        return -p, p
    
    def init_all(self, q0):
        for qi in q0:
            self.init(*qi)
    
    def perform_selected(self, q0, N, dt=0.01, err=1e-8):
        q0 = [[*qi] for qi in q0]
        for qi in q0:
            self.orbit[*qi].pintegrate(N, dt, err)
    
    def perform_all(self, N, dt=0.01, err=1e-8):
        self.perform_selected(list(self.orbit), N, dt, err)
    

    def periodic_orbits(self, n, xmin, xmax, Nsearch, xerr = 1e-8, xtol=1e-3, dt=1e-2, err=1e-8):
        def px(x0):
            orb = self.new_orbit(x0, 0)
            orb.pintegrate(n, dt, err)
            return orb.px[-1]
        
        def isroot(x):
            orb = self.new_orbit(x, 0)
            orb.pintegrate(n, dt, err)
            return all([abs(xi-x)>xtol for xi in orb.x[1:-1]]) and abs(x-orb.x[-1])<xtol
        
        ps = self.newcopy(load=False)
        all_intervals = [[x0, 0] for x0 in np.linspace(xmin, xmax, Nsearch)]
        ps.init_all(all_intervals)
        ps.perform_all(n, dt, err)

        q = list(ps.orbit)
        roots = []
        for i in range(Nsearch-1):
            if ps.orbit[q[i]].px[-1] * ps.orbit[q[i+1]].px[-1] < 0:
                x1, x2 = all_intervals[i][0], all_intervals[i+1][0]
                x0 = bisect(px, x1, x2, xerr)
                if isroot(x0):
                    roots.append(x0)
        return roots

    def periodic_orbit_near(self, x0, px0, n, derr=1e-8, dt=1e-2, err=1e-8)->tuple[float, float]:
        
        def dist(q):
            try:
                x, px = q
                orb = self.new_orbit(x, px)
                orb.pintegrate(n, dt, err)
                return np.array([orb.x[-1]-x, orb.px[-1]-px])
            except:
                return np.array([1e10, 1e10])

        return fsolve(dist, [x0, px0], xtol=derr)

    def rotation_number(self, xobs, pxobs, x0, px0, N=300, ode={}):
        orb = self.new_orbit(x0, px0)
        orb.pintegrate(N, **ode)
        x = [xi-xobs for xi in orb.x]
        px = [pxi-pxobs for pxi in orb.px]
        z = [xi+1j*pxi for xi, pxi in zip(x, px)]
        n = len(z)-1
        Deltaphi = [np.angle(z[i+1]/z[i]) for i in range(n)]
        return sum(Deltaphi)/(2*np.pi*n)
    
    def full_rot_num(self, xobs, pxobs, a, b, split=200, N=300, axis='x', ode={}):
        arr = np.linspace(a, b, split)
        if axis == 'x':
            q = [[xi, pxobs] for xi in arr]
        elif axis == 'px':
            q = [[xobs, pxi] for pxi in arr]
        ps = self.newcopy(load=False)
        ps.init_all(q)
        ps.perform_selected(q, N, **ode)
        rot = []
        arr_final = []
        for j, qi in enumerate(q):
            orb = ps.orbit[*qi]
            x = [xi-xobs for xi in orb.x]
            px = [pxi-pxobs for pxi in orb.px]
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
        for orb in self:
            xarr, yarr = np.array([orb.x[0], orb.px])
            d = np.min((xarr-x)**2 + (yarr-px)**2)
            if d < _min:
                _min = d
                _orb = orb
        return _orb

    def nearest_item(self, x, px):
        _min = np.inf
        for art in self.figure.artists:
            xarr, yarr = np.array([art.xcoords, art.ycoords])
            if xarr.flatten().shape != (0,):
                d = np.min((xarr-x)**2 + (yarr-px)**2)
                if d < _min:
                    _min = d
                    _art = art
        return _art

    def _draw(self, lims='current'):
        ax = self._ax
        if lims == 'current':
            xlims = ax.get_xlim()
            ylims = ax.get_ylim()
        else:
            xlims = (None, None)
            ylims = (None, None)
        ax.clear()
        self.figure.draw(ax)
        ax.set_xlim(*xlims)
        ax.set_ylim(*ylims)
        self._fig.canvas.draw_idle()

    def handle(self, event: MouseEvent, N: int, **kwargs):
        x, px = event.xdata, event.ydata
        if event.button == 1:
            if event.key is None:
                self.temp.add(x, px)
            elif event.key == 'delete':
                for q in self.orbit.copy():
                    if self.orbit[q].is_active:
                        self.orbit.pop(q)
                self.temp.clear()
            elif event.key == 'control':
                self.nearest_orbit(x, px)._set(is_active=True)
            elif event.key == 'shift':
                self.temp.clear()
                self.deactivate_all()
            elif event.key == 'ctrl+shift':
                orb = self.nearest_orbit(x, px)
                x, y = interpolate_line(*reorder_coords(orb.x, orb.px))
                self.figure._artists.append(self.proj.line(x=x, y=y, c='red', zorder=4))
                self.figure._artists.append(self.proj.line(x=x, y=-np.array(y), c='red', zorder=4))
        elif event.button == 3 and event.key is None:
            if not self.temp.isempty():
                self.init_all(self.temp.all_coords())
                for q in self.temp.all_coords():
                    self.orbit[q]._set(is_active=True)
                self.temp.clear()

            active_orbs = [q for q in self.orbit if self.orbit[q].is_active]
            if active_orbs:
                self.perform_selected(active_orbs, N, **kwargs)

        self._draw()

    def plot(self, interact = False, N=200, ode_args={}):
        fig: Fig
        ax: Axes
        fig, ax = plt.subplots(figsize=(5, 5))
        self._ax = ax
        self._fig = fig

        self._draw('auto')
        if interact:
            fig.canvas.mpl_connect('button_press_event', lambda event: self.handle(event, N, **ode_args))
            
        return fig, ax

    def remove_divergent(self):
        for q in list(self.orbit):
            if self.orbit[q].diverges:
                self.remove(*q)

    def flag_all(self):
        for q in self.orbit:
            self.orbit[q]._set(is_flagged = not self.orbit[q].is_flagged)


class Henon_Heiles(PoincareSection):

    def __init__(self, eps, alpha=None, beta=0, gamma=0, omega_x=1., omega_y=1., load=True, project_folder=os.getcwd()):
        if alpha is None:
            alpha = Rat(-1, 3) #standard Henon_Heiles implementation
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega_x = omega_x
        self.omega_y = omega_y
        self._args = (eps, alpha, beta, gamma, omega_x, omega_y)
        super().__init__(project_folder)
        if load:
            folder = os.path.join(project_folder, 'PoincareSections')
            file = os.path.join(folder, f"{self.name}_data.txt")
            if os.path.exists(folder):
                try:
                    data = tools.try_read(file, error=True)
                    for stuff in data:
                        orb = self.init(*stuff['_q'][0, [1, 3]])
                        orb.set_state(**stuff)
                    fig = SquareFigure.load(os.path.join(project_folder, 'figdata'), self.name)
                    self.figure = PSFig(self, **fig.parameters)
                    self.figure._artists = fig.artists
                except FileNotFoundError:
                    pass
        self.figure._artists.insert(0, self.proj.scatter(x=[], y=[], c='forestgreen'))
        self.project_folder = project_folder

    def V(self, x, y):
        return 1/2*(self.omega_x**2*x**2 + self.omega_y**2*y**2) + self.eps*(x*y**2 + self.alpha*x**3 + self.beta*x**2*y + self.gamma*y**3)
    
    @property
    def title(self):
        return r'$\epsilon$={0}, $\alpha$={1}, $B = {2}$'.format(self.eps, self.alpha, round(self.omega_y**2, 5))

    @property
    def name(self):
        A = round(self.omega_x**2, 10)
        B = round(self.omega_y**2, 10)
        return f'{self.eps}'.replace('.', '').replace('/', '_') + '__' + f'{self.alpha}'.replace('.', '').replace('/', '_') + '__' + f'{A}'.replace('.', '').replace('/', '_')+'__' + f'{B}'.replace('.', '').replace('/', '_')

    def intervals(self, y=0., px=0.):
        '''
        we demand that T = E - V(x, y=y) >= 0
        '''
        def f(x):
            return self.E - self.V(x, y) - px**2/2
        kappa = self.alpha*self.eps
        if kappa == 0:
            xc = -self.eps*y**2/self.omega_x**2
            if f(xc) <= 0:
                return ()
            else:
                return sorted(np.roots([-self.omega_x**2/2, -self.eps*y**2, self.E-1/2*self.omega_y**2*y**2-1/2*px**2]).real),

        roots = roots_of_poly3(-kappa, -self.omega_x**2/2, -self.eps*y**2, self.E-1/2*self.omega_y**2*y**2-1/2*px**2)

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
            raise ValueError(f'Infinite interval for eps={self.eps}, alpha={self.alpha}: {intervals}')
        for inter in intervals:
            if not np.isinf(inter).any():
                xmin, xmax = inter
                break
        return xmin, xmax


    def draw_shell(self, boundary=None, n=1000):
        try:
            xmin, xmax = self.region()
            exists_reg = True
        except:
            exists_reg = False
        if boundary is not None or not exists_reg:
            full = sum(self.intervals(), start=[])
            xmin, xmax = np.min(full), np.max(full)
            if xmin == -np.inf:
                xmin = boundary
            elif xmax == np.inf:
                xmax = boundary
            assert xmin is not None and xmax is not None
        # r = roots_of_poly3(2*self.kappa, self.omega_x**2, 0, -2)
        x = np.linspace(xmin+1e-6, xmax-1e-6, n)
        x_final = []
        px = []
        for xi in x:
            try:
                pxmin, pxmax = self.px_limits(xi)
                x_final.append(xi)
                px.append(pxmax)
            except AssertionError:
                continue
        x = np.array(x_final)
        px =  np.array(px)

        xs, pxs = split_parts(x, px, d=0.1)
        for i, (xi, pxi) in enumerate(zip(xs, pxs)):
            x = np.append(xi, xi[::-1])
            px = np.append(pxi, -pxi[::-1])
            self.figure._artists.append(self.proj.line(x=x, y=px, c='red', zorder=4))

    def split_interval(self, n=30):
        xmin, xmax = self.region()
        return [[x0, 0] for x0 in np.linspace(xmin+1e-6, xmax-1e-6, n)]

    def init_axis(self, n=30):
        q = self.split_interval(n)
        self.init_all(q)

    def periodic_orbits(self, n, xmin=None, xmax=None, Nsearch=200, xerr = 1e-8, xtol=1e-3, dt=1e-2, err=1e-8):
        if xmin is None:
            xmin = self.intervals()[0][0] + 1e-6
            assert not np.isinf(xmin)
        elif xmax is None:
            xmax = self.intervals()[-1][-1] - 1e-6
            assert not np.isinf(xmax)
        elif xmin is None and xmax is None:
            xmin1, xmax1 = self.region()
        if xmin is None:
            xmin = xmin1+1e-6
        if xmax is None:
            xmax = xmax1-1e-6
        return super().periodic_orbits(n, xmin, xmax, Nsearch, xerr, xtol, dt, err)

    def full_rot_num(self, xobs, pxobs, split=200, N=300, axis='x', ode={}):
        if axis == 'x':
            a, b = self.region()
        elif axis == 'px':
            a, b = self.px_limits(x=xobs, y=0)
        a = a+1e-6
        b = b-1e-6
        return super().full_rot_num(xobs, pxobs, a, b, split, N, axis=axis, ode=ode)

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

    def remove_divergent(self):
        for q in list(self.orbit):
            if self.orbit[q].diverges:
                del self.orbit[q]

    def get_limiting_px(self, pxmin, pxmax, x=0., tol=1e-6, err=1e-8, N=100):

        def f(px):
            orb = self.new_orbit(x, px)
            orb.pintegrate(N, err=err)
            return orb.diverges - 0.5
        
        return Bisect(f, pxmin, pxmax, tol=tol)
    
    def save(self, xlims=None, ylims=None, proj=True):
        ps_folder = os.path.join(self.project_folder, 'PoincareSections')
        if not os.path.exists(ps_folder):
            os.makedirs(ps_folder)

        tools.write_binary_data(os.path.join(ps_folder, f"{self.name}_data.txt"), [orb.get_state() for orb in self])

        fig = self.figure.copy()
        fig._artists.pop(0)
        if xlims is not None:
            fig.xlims = xlims
        if ylims is not None:
            fig.ylims = ylims
        if proj:
            self.proj.savefig(fig)
        else:
            fig.save(self.project_folder)

    def move_to(self, folder):
        ps = self.__class__(*self._args, load=False, project_folder=folder)
        ps.figure = self.figure.copy()
        ps.orbit = {tup: self.orbit[tup].copy() for tup in self.orbit}
        ps.save()


class Integrable_Henon_Heiles(Henon_Heiles):

    cases = {'1/3': Rat(1, 3), '2': 2, '16/3': Rat(16, 3)}

    def __init__(self, eps, alpha: Literal['1/3', '2', '16/3'], omega_x=None, omega_y=None, load=True, project_folder=os.getcwd()):
        alpha = str(alpha)
        self._alpha = alpha
        if alpha != '16/3':
            omega_x = 1 if omega_x is None else omega_x
            omega_y = 1 if omega_y is None else omega_y
        else:
            if omega_x is None and omega_y is None:
                omega_y = 1
                omega_x = 4
            elif omega_x is None:
                omega_x = 4*omega_y
            elif omega_y is None:
                omega_y = omega_x/4
            else:
                assert omega_x == 4*omega_y
        super().__init__(eps=eps, alpha=self.cases[alpha], omega_x=omega_x, omega_y=omega_y, load=load, project_folder=project_folder)

    def _K(self, x, y, p_x, p_y):
        if self._alpha == '1/3':
            return p_x*p_y + x*y + self.eps*(y*x**2 + y**3/3)
        elif self._alpha == '2':
            A, B = self.omega_x**2, self.omega_y**2
            Lxy = x*p_y - y*p_x
            return -4*Lxy*p_y + (4*B*x+self.eps*y**2+4*self.eps*x**2)*y**2+(4*B-A)/self.eps*(p_y**2 + B*y**2)
        elif self._alpha == '16/3':
            A, B = self.omega_x**2, self.omega_y**2 #A = 16*B already satisfied
            return p_y**4 + 2*(B+2*self.eps*x)*y**2*p_y**2-4/3*self.eps*y**3*p_x*p_y+B**2*y**4-4/3*self.eps*(B+self.eps*x)*y**4*x-2/9*self.eps**2*y**6
        
    def K(self, x, p_x):
        py2 = 2*(self.E-self.eps*self.alpha*x**3)-self.omega_x**2*x**2-p_x**2 # y = 0
        if self._alpha == '1/3':
            return p_x*py2**0.5
        elif self._alpha == '2':
            A, B = self.omega_x**2, self.omega_y**2
            return ((4*B-A)/self.eps-4*x)*py2
        elif self._alpha == '16/3':
            return py2**2

    def dKdx_roots(self)->list[float]:
        assert self._alpha == '2'
        A, B = self.omega_x**2, self.omega_y**2
        l = (4*B-A)/self.eps
        return roots_of_poly3(32*self.eps*self.alpha, 3*(4*A-2*l*self.eps*self.alpha), -2*l*A, -8*self.E)
    
    def per_orbits(self)->list[float]:
        assert self._alpha == '2'
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
        assert self._alpha == '2'
        assert len(self.intervals()) == 1
        res = []
        for x in self.per_orbits():
            x1 = self.get_limiting_x(-5, x, n=2)
            x2 = self.get_limiting_x(x, n=2)
            res.append((x1, x, x2))
        return res


    def Py2(self, x, k):
        if self._alpha == '2':
            return k/(-4*x+(4*self.omega_y**2-self.omega_x**2)/self.eps)
        elif self._alpha == '16/3':
            return k**0.5+0*x
        
    def Px2(self, x, k):
        if self._alpha == '1/3':
            a, b, c = 1, self.omega_x**2*x**2 + 2*self.eps*self.alpha*x**3 - 2*self.E, k**2
            D = b**2 - 4*a*c
            return (-b + np.power(D, 0.5, dtype=complex))/(2*a), (-b - np.power(D, 0.5, dtype=complex))/(2*a)
        else:
            return 2*self.E-self.Py2(x, k) - self.omega_x**2*x**2 - 2*self.eps*self.alpha*x**3

    def newcopy(self, load=False):
        return Integrable_Henon_Heiles(self.eps, self._alpha, self.omega_x, self.omega_y, load=load, project_folder=self.project_folder)
    
    def draw_k(self, xmin, xmax, px=0., n=1000, **kwargs):
        fig = self.proj.figure('K_'+self.name, self.title, 'x', 'K', yrot=0, square=True)
        fig.aspect = 'auto'
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

    
    def add_invariant(self, k, xmin, xmax, n=1000, **kwargs):
        x = np.linspace(xmin, xmax, n)
        if self._alpha != '1/3':
            px2 = self.Px2(x, k)
            mask = np.logical_and(np.isreal(px2), np.logical_and(self.Py2(x, k)>=0, px2>=0))
            px2 = px2[mask]
            x = x[mask]
        else:
            px2_up, px2_down = self.Px2(x, k)
            px2 = np.array([*px2_up, *px2_down[::-1]])
            x = np.array([*x, *x[::-1]])
            mask = np.logical_and(np.isreal(px2), px2 >= 0)
            x = x[mask]
            px2 = px2[mask]
        
        if self._alpha != '1/3':
            x, px = split_parts(x, px2**0.5, d=0.1)
            label = kwargs.pop('label', '')

            for j, (xi, pxi) in enumerate(zip(x, px)):
                # xi, pxi = [*xi, *xi], [*pxi, *(-pxi)]#reorder_coords([*xi, *xi], [*pxi, *(-pxi)])
                xi, pxi = reorder_coords([*xi, *xi], [*pxi, *(-pxi)])
                xi = [*xi, xi[0]]
                pxi = [*pxi, pxi[0]]
                if j == 0 and label != '':
                    self.figure.add(self.proj.line(xi, pxi, **kwargs, label=label))
                else:
                    self.figure.add(self.proj.line(xi, pxi, **kwargs))
        else:
            x, px = split_parts(x, px2**0.5)
            label = kwargs.pop('label', '')

            for i in range(2):
                for j, (xi, pxi) in enumerate(zip(x, px)):
                    if j == 0 and i == 0 and label != '':
                        self.figure.add(self.proj.line(xi, (-1)**i*pxi, **kwargs, label=label))
                    else:
                        self.figure.add(self.proj.line(xi, (-1)**i*pxi, **kwargs))

    def zero_vel_curve(self, xmin, xmax, n = 10000, **kwargs):
        x = np.linspace(xmin, xmax, n)
        xcrit = -self.omega_y**2/(2*self.eps)
        r = roots_of_poly3(-2*self.eps*self.alpha, -self.omega_x**2, 0, 2*self.E)
        r.append(xcrit)
        r.sort()
        fig = self.proj.figure('czv_'+self.name, self.title, 'x', 'y', yrot=0, square=True)
        for i, ri in enumerate(r[:-1]):
            x = np.linspace(r[i], r[i+1], n)
            y2 = (2*self.E-self.omega_x**2*x**2-2*self.eps*self.alpha*x**3)/(self.omega_y**2+2*self.eps*x)
            mask = y2 >= 0
            x = x[mask]
            y = y2[mask]**0.5
            if ri == xcrit:
                x = [*x, *x[::-1]]
                y = [*y, *(-y)[::-1]]
            elif r[i+1] == xcrit:
                x = [*x[::-1], *x]
                y = [*y[::-1], *(-y)]
            else:
                x = [*x, *x[::-1]]
                y = [*y, *(-y)[::-1]]
                x = [*x, x[0]]
                y = [*y, y[0]]
            fig.add(self.proj.line(x, y, **kwargs))

        return fig

    def get_zero_vel_curve_roots(self):
        try:
            x1, x2, x3 = roots_of_poly3(-2*self.eps*self.alpha, -self.omega_x**2, 0, 2*self.E)
        except ValueError:
            return roots_of_poly3(-2*self.eps*self.alpha, -self.omega_x**2, 0, 2*self.E)
        if self.alpha >= 0:
            return x3, x2, x1
        else:
            return x1, x2, x3

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
                if (ps.orbit[(x[i], 0)].diverges-0.5) * (ps.orbit[(x[j], 0)].diverges-0.5) < 0:
                    return bisectright(f, x[i], x[j], tol=1e-5)
        raise ValueError('Bisection limits invalid')

        


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
    
def when_perorb_disappears(eps, a_min, a_max, xmin, tol=1e-6, **kwargs):
    def f(a):
        ps = Henon_Heiles(eps, a)
        if len(ps.periodic_orbits(1, xmin=xmin, Nsearch=10, **kwargs)) > 0:
            return 1
        else:
            return -1
    
    return bisect(f, a_min, a_max, tol=tol)

def split_parts(x: np.ndarray, *arrs: np.ndarray, d=1):
    y_arrs = [[] for _ in arrs]
    dx = np.diff(x)
    dy = np.array([abs(np.diff(arr)) for arr in arrs])
    dy = dy.max(axis=0)
    assert dy.shape == dx.shape, dy.shape
    
    split = np.argwhere(abs(dx**2 + dy**2)>d**2).flatten()+1
    if len(split) > 0:
        x_arrs = np.split(x, split)
        y_arrs = [np.split(arri, split) for arri in arrs]
    else:
        x_arrs = [x]
        y_arrs = [[yarri] for yarri in arrs]
    return x_arrs, *y_arrs


def positive_intervals(*polys: list[float]):
    from sympy import abc, real_roots
    x = abc.x
    res = 1
    for poly in polys:
        n = len(poly)-1
        res = res * sum([poly[i]*x**(n-i) for i in range(n+1)])
    
    real_roots(res)

def interpolate_line(x, y):
    from scipy.interpolate import splprep, splev
    # Ensure input is numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Create a parameterized curve
    points = np.array([x, y])
    tck, u = splprep(points, s=0)  # 's=0' gives interpolation through all points

    # Generate new parameter values for finer interpolation (3 times the original points)
    u_fine = np.linspace(0, 1, len(x) * 3)
    x_new, y_new = splev(u_fine, tck)
    return x_new, y_new


def reorder_coords(x, y):
    x = np.array(x)
    y = np.array(y)
    points = np.column_stack((x, y))
    
    # Start from the first point and initialize the ordered list
    ordered_points = [points[0]]
    remaining_points: np.ndarray = points[1:]
    
    while len(remaining_points) > 0:
        # Find the distance from the last point in ordered_points to each remaining point
        distances = np.linalg.norm(remaining_points - ordered_points[-1], axis=1)
        
        # Find the index of the nearest point
        nearest_index = np.argmin(distances)
        
        # Add the nearest point to ordered_points and remove it from remaining_points
        ordered_points.append(remaining_points[nearest_index].copy())
        remaining_points = np.delete(remaining_points, nearest_index, axis=0)
    
    # Split the ordered points back into x and y arrays
    ordered_points = np.array(ordered_points)
    ordered_x, ordered_y = ordered_points[:, 0], ordered_points[:, 1]
    
    return ordered_x, ordered_y

