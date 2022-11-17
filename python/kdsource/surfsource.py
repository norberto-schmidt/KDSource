import os
import subprocess
from enum import Enum
from math import cos, pi, sin

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.stats import knuth_bin_width
from matplotlib.colors import LogNorm
from uncertainties import ufloat

import mcpl

plt.style.use(os.path.dirname(os.path.abspath(__file__))+'/plots.mplstyle')


MCPLColumns = ['id', 'type', 'E', 'x', 'y', 'z', 'u', 'v', 'w',
               't', 'wgt', 'px', 'py', 'pz', 'userflags']
XUNITS = {'x': 'cm', 'y': 'cm', 'z': 'cm', 'R': 'cm', 'theta': '$^o$',
          'u': '', 'v': '', 'w': '', 'phi': '$^o$', 'psi': '$^o$', 'mu': '',
          'E': 'MeV', 'ln(E0/E)': '', 'lambda': '$\AA$',
          't': 'ms', 'log(t)': 'ms'}
YUNITS = {'x': 'cm$^{-1}$', 'y': 'cm$^{-1}$', 'z': 'cm$^{-1}$', 'R': 'cm$^{-2}$', 'theta': 'rad$^{-1}$',
          'u': '', 'v': '', 'w': '', 'phi': 'rad$^{-1}$', 'psi': 'rad$^{-1}$', 'mu': '',
          'E': 'eV$^{-1}$', 'ln(E0/E)': '', 'lambda': '$\AA^{-1}$',
          't': 's$^{-1}$', 'log(t)': ''}
XLATEX = {'x': 'x', 'y': 'y', 'z': 'z', 'R': 'R', 'theta': r'\theta',
          'u': 'u', 'v': 'v', 'w': 'w', 'phi': r'\varphi', 'psi': r'\vartheta', 'mu': r'\mu=\cos(\vartheta)',
          'E': 'E', 'lambda': '\lambda', 'ln(E0/E)': r'\ln(E_0/E)', 'lambda': r'\lambda',
          't': 't', 'log(t)': r'\log(t)'}
YSCALE = {'x': 1, 'y': 1, 'z': 1, 'R': 1, 'theta': 1,
          'u': 1, 'v': 1, 'w': 1, 'phi': 1, 'psi': 1, 'mu': 1,
          'E': 1e6, 'ln(E0/E)': 1, 'lambda': 1,
          't': 1e-3, 'log(t)': 1}

class PDGCode(Enum):
    NEUTRON = 2112
    PHOTON = 22
    ELECTRON = 11
    POSITRON = -11
    PROTON = 2212

class OpenMCCode(Enum):
    NEUTRON = 0
    PHOTON = 1
    ELECTRON = 2
    POSITRON = 3
    PROTON = 4

class SurfaceSourceFile:
    """ Read a surface source file

    Parameters
    ----------
    filepath: str
        Path to surface source file
    S0: float
        Simulation source factor
        Default: 1.0
    Nparticles: int
        Total source particles simulated
        Default: 1e9
    translation: list
        Translation for the position variables [cm]
        Default is [0, 0, 0]
    rotation: list
        Rotation for the position and direction variables [deg]
        Default is [0, 0, 0]
    domain: dict
        Range of the variables: var_min <= 'var' <= var_max
        It must be defined like: {'var':[var_min, var_max]}
        List of possible variables:
        type [PDG],
        E [MeV], ln(E0/E), lambda [$\AA$]
        x [cm], y [cm], z [cm], R [cm], theta [rad],
        u, v, w, phi [rad], psi [rad],
        t [ms], log(t),
        wgt
        Default is {}
    set_domain_first: bool
        Define if the variables domains and the normal direction
        must be set before or after the translation and rotation 
        Default is False 
    set_rotation_first: bool
        Define if the rotation must be done before or after
        the translation of the coordinates
        Default is False
    E0: float
        Reference energy to calculate the lethargy ln(E0/E)
        Default is 20 MeV
    uvw_reference: str
        Versor which is normal to the surface source after the rotation
        Default is 'w'
    surface: int
        Surface id 
    Nmax: int
        Max
    """

    def __init__(self, filepath, S0=1.0, Nparticles=1e9, dA=None, translation=[0, 0, 0], rotation=[0, 0, 0],
                 domain={}, set_domain_first=False, set_rotation_first=False, E0=20, convoluted=False,
                 uvw_reference='w', surface=None, Nmax=1e9, current=None, tpulse=None,):
        # Initialize SurfaceSource class atributes
        self._filepath = filepath
        self._extension = os.path.splitext(self._filepath)[-1]
        self._translation = translation
        self._rotation = rotation
        self._domain = domain
        self._domain_first = set_domain_first
        self._rotation_first = set_rotation_first
        self._E0 = E0
        self._uvw = uvw_reference
        self._S0 = S0
        self._Np = int(Nparticles)
        self._dA = dA
        self._surface = surface # if surface == None else surface
        self._max_particles = int(Nmax)
        self._tpulse = tpulse
        self._current = current
        self._convoluted = convoluted
        self._df = self.__read__()
        self._df2 = self.get_pandas_dataframe()

    def __read__(self):
        # OpenMC .h5 format
        if self._extension == '.h5':
            with h5py.File(self._filepath, 'r') as fh:
                df = pd.DataFrame(columns=MCPLColumns)
                # Change from OpenMC ParticleType type to MCPL PDGCode
                df['type'] = fh['source_bank']['particle']
                for pin, pout in zip((OpenMCCode), (PDGCode)):
                    df.loc[df['type'] == pin.value, 'type'] = pout.value
                df['id'] = df.index
                df['E'] = fh['source_bank']['E']*1e-6  # eV to MeV
                df['x'] = fh['source_bank']['r']['x']  # cm
                df['y'] = fh['source_bank']['r']['y']  # cm
                df['z'] = fh['source_bank']['r']['z']  # cm
                df['u'] = fh['source_bank']['u']['x']
                df['v'] = fh['source_bank']['u']['y']
                df['w'] = fh['source_bank']['u']['z']
                df['wgt'] = fh['source_bank']['wgt']
                df['t'] = fh['source_bank']['time']*1e3  # s to ms
                df['px'] = 0.0
                df['py'] = 0.0
                df['pz'] = 0.0
                df['userflags'] = 0x00000000
        # MCPL format
        elif self._extension == '.gz' or self._extension == '.mcpl':
            with mcpl.MCPLFile(self._filepath, blocklength=self._max_particles) as mcpl_file:
                pb = mcpl_file.read_block()
                plist = np.array([pb.pdgcode, pb.ekin, pb.x, pb.y, pb.z, pb.ux, pb.uy, pb.uz,
                                  pb.time, pb.weight, pb.polx, pb.poly, pb.polz, pb.userflags])
                df = pd.DataFrame(plist.T, columns=MCPLColumns[1:])
                if self._surface is not None:
                    df = df[df['userflags'] == int(self._surface)]
                df['id'] = np.arange(0, len(df))
                df = df[MCPLColumns]
        # ASCII format, for all the other cases
        else:
            df = pd.DataFrame(np.loadtxt(
                self._filepath, skiprows=5), columns=MCPLColumns)
        # Print some info regarding the particles in the file
        print('Number of particles in file {:s}: {:d}'.format(
            self._filepath, len(df)))
        for p, idx in zip((PDGCode), ['n', 'g', 'e-', 'e+', 'p']):
            print('{:.0f}% {:s}, '.format(
                len(df[df['type'] == p.value])/len(df['type'])*100, idx), end='')
        print('')
        return df

    def get_pandas_dataframe(self):
        """" Obtain a DataFrame with the particles that were saved in the file        
        """
        df = self._df.copy()
        # Define the coordinates system
        if self._uvw == 'u':
            X, Y, Z, U, V, W = 'y', 'z', 'x', 'v', 'w', 'u'
        elif self._uvw == 'v':
            X, Y, Z, U, V, W = 'z', 'x', 'y', 'w', 'u', 'v'
        elif self._uvw == 'w':
            X, Y, Z, U, V, W = 'x', 'y', 'z', 'u', 'v', 'w'
        self._X = X
        self._Y = Y
        self._Z = Z
        # Check domain
        if self._domain_first:
            df = self._set_domain(df)
        # Translate and rotate the coordinates
        if self._rotation_first:
            df = self._set_rotation(df)
            df = self._set_translation(df)
        else:
            df = self._set_translation(df)
            df = self._set_rotation(df)
        # Normalize the direction
        self._normalize_direction(df)
        # Convolute time considering the pulse width
        if self._tpulse != None and self._convoluted == False:
            self._convolute(df, {'t': self._tpulse * 1e3})  # from s to ms
            self._convoluted = True
        # Calculate polar coordinates, spherical angles, lethargy, time-decade and wavelength
        df['R'] = (df[X].to_numpy()**2+df[Y].to_numpy()**2)**0.5
        df['theta'] = (np.arctan2(df[Y].to_numpy(), df[X].to_numpy()))
        df['mu'] = df[W].to_numpy()
        df['psi'] = (np.arccos(df[W].to_numpy()))
        df['phi'] = (np.arctan2(df[V].to_numpy(), df[U].to_numpy()))
        df['ln(E0/E)'] = np.log(self._E0/df['E'].to_numpy())
        df['log(t)'] = np.log10(df['t'].to_numpy())
        df['lambda'] = 72.3/252.8*(df['E']*1e6)**-0.5
        # Check domain
        if not self._domain_first:
            df = self._set_domain(df)
        return df

    def _set_translation(self, df):
        # Translate the coordinates (only position)
        df['x'] = df['x'].to_numpy() + self._translation[0]
        df['y'] = df['y'].to_numpy() + self._translation[1]
        df['z'] = df['z'].to_numpy() + self._translation[2]

        return df

    def _set_rotation(self, df):
        # Rotate the coordinates (posisition and direction)
        phi, theta, psi = np.array(self._rotation)*(pi/180.)
        c3, s3 = cos(phi), sin(phi)
        c2, s2 = cos(theta), sin(theta)
        c1, s1 = cos(psi), sin(psi)
        rotation_matrix = np.array([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2],
                                    [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3],
                                    [-s2, c2*s3, c2*c3]])
        df['x'], df['y'], df['z'] = rotation_matrix @ np.array(
            [df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()])
        df['u'], df['v'], df['w'] = rotation_matrix @ np.array(
            [df['u'].to_numpy(), df['v'].to_numpy(), df['w'].to_numpy()])
        return df

    def _normalize_direction(self, df):
        # Normalize the direction versor
        unorm = df['u'].to_numpy()**2 + df['v'].to_numpy()**2 + \
            df['w'].to_numpy()**2

        df['u'], df['v'], df['w'] = (df['u'].to_numpy()/unorm**0.5,
                                     df['v'].to_numpy()/unorm**0.5,
                                     df['w'].to_numpy()/unorm**0.5)
        return df

    def _set_domain(self, df):
        # Check the filters coordinates
        for pvar, (pmin, pmax) in self._domain.items():
            if pvar == 'psi' or pvar == 'phi' or pvar == 'theta':
                pmin, pmax = np.deg2rad([pmin, pmax])
            if pmin != None:
                df = df[df[pvar] >= pmin]
            if pmax != None:
                df = df[df[pvar] <= pmax]
        return df

    def _convolute(self, df, convolution):
        # Convolute the desired variables
        # Set the seed to reproduce the results
        np.random.seed(len(df))
        for cvar, cwidth in convolution.items():
            df[cvar] = df[cvar].to_numpy() + np.random.uniform(low=0.0,
                                                               high=cwidth, size=len(df[cvar]))
        return df

    def _get_brilliance(self, df, norm_vars):
        self._brilliance = 1.0
        Sunit = self._get_units(norm_vars)

        if self._dA is not None:
            self._brilliance /= self._dA

        if 'mAs' in norm_vars:
            self._brilliance /= (1.6022e-19*1e3)
            if self._current != None:
                self._brilliance *= self._current
            if self._tpulse != None:
                self._brilliance *= self._tpulse
            norm_vars = [x for x in norm_vars if x not in ['mAs']]

        for nvar in norm_vars:
            pmin = min(df[nvar]) * YSCALE[nvar]
            pmax = max(df[nvar]) * YSCALE[nvar]
            if nvar == 'R':
                dp = 0.5*(pmax**2 - pmin**2)
            elif nvar == 'psi':
                dp = -(np.cos(pmax) - np.cos(pmin))
            else:
                dp = pmax - pmin

            self._brilliance /= dp

        return Sunit

    def _get_units(self, norm_vars):
        Sunit = ''
        dA = False
        dOmega = False

        if 'mAs' in norm_vars:
            if self._current == None and self._tpulse == None:
                Sunit += '(mA s)$^{-1}$ '
            elif self._current == None:
                Sunit += 'mA$^{-1}$ '
            elif self._tpulse == None:
                Sunit += 's$^{-1}$ '
            else:
                if 't' not in norm_vars:
                    Sunit += 'pulse$^{-1}$ '
            norm_vars = [x for x in norm_vars if x not in ['mAs']]

        if self._dA is not None:
            Sunit += 'cm$^{-2}$ '
            dA = True

        if 'x' in norm_vars and 'y' in norm_vars and dA == False:
            Sunit += 'cm$^{-2}$ '
            norm_vars = [x for x in norm_vars if x not in ['x', 'y']]

        if 'x' in norm_vars and 'z' in norm_vars and dA == False:
            Sunit += 'cm$^{-2}$ '
            norm_vars = [x for x in norm_vars if x not in ['x', 'z']]

        if 'y' in norm_vars and 'z' in norm_vars and dA == False:
            Sunit += 'cm$^{-2}$ '
            norm_vars = [x for x in norm_vars if x not in ['y', 'z']]

        if 'R' in norm_vars and 'theta' in norm_vars and dA == False:
            Sunit += 'cm$^{-2}$ '
            norm_vars = [x for x in norm_vars if x not in ['R', 'theta']]

        if 'mu' in norm_vars and 'phi' in norm_vars and dOmega == False:
            Sunit += 'sr$^{-1}$ '
            norm_vars = [x for x in norm_vars if x not in ['mu', 'phi']]
            dOmega = True

        if 'psi' in norm_vars and 'phi' in norm_vars and dOmega == False:
            Sunit += 'sr$^{-1}$ '
            norm_vars = [x for x in norm_vars if x not in ['psi', 'phi']]
            dOmega = True

        for nvar in norm_vars:
            Sunit += '{:s} '.format(YUNITS[nvar])

        return Sunit

    def _get_info(self, df, vars):
        info = 'Surface: {:d}, N: {:d}\n'.format(self._surface if self._surface!=None else 0, len(df))
        info += 'S0: {:.2e}\n'.format(self._S0)
        info += 'Nsim: {:.2e}\n'.format(self._Np)
        info += 'Area: {:.2e} cm$^2$\n'.format(
            self._dA if self._dA is not None else 0.0)
        # info += 'Translation: [{:.1f}, {:.1f}, {:.1f}]\n'.format(self._translation[0], self._translation[1], self._translation[2])
        # info += 'Rotation: [{:.1f}, {:.1f}, {:.1f}]\n\n'.format(self._rotation[0], self._rotation[1], self._rotation[2])
        info += '\n'
        info += 'Brilliance:\n'
        for pvar in vars:
            if pvar != 'mAs':
                pmin = min(df[pvar])
                pmax = max(df[pvar])
                if pvar == 'theta' or pvar == 'psi' or pvar == 'phi':
                    pmin, pmax = np.rad2deg([pmin, pmax])
                info += '${:s}$: [{:.2e}, {:.2e}] {:s}'.format(
                    XLATEX[pvar], pmin, pmax, XUNITS[pvar])+'\n'
            else:
                if self._current != None:
                    info += 'Current: {:.2f} mA\n'.format(self._current)
                if self._tpulse != None:
                    info += 'Pulse width: {:.2f} $\mu$s\n'.format(
                        self._tpulse*1e6)
        return info

    def get_distribution(self, vars, bins, scales=['linear'], factor=1.0, filters={}, norm_vars=[], total=False, convolution=None):
        """ Obtain a N-D distribution for given variables

        Parameters
        ----------
        var: list of str
            Variables to calculate the distribution
        bins: list of int or list of iterable
            Bins
            If int, then the distributions are going to be calculated between
            min(var), max(var) for int bins
        factor: float
            Multiply the distribution by a specific numerical factor
            Default: 1.0
        filters: dict
            Specify the other coordinates domain for the distribution
            It must be defined like: {'var':[var_min, var_max]}
            Default: {}
        total: bool
            Define if the integral value is required
            Default: False
        norm_vars: list
            List of string with the variables to calculate the brilliance

        Returns
        -------

        """
        pinfo = ''
        old_filters = self._domain
        self._domain = filters
        # df = self.get_pandas_dataframe()
        df = self._df2.copy()
        df = self._set_domain(df)
        self._domain = old_filters

        # Normalize the distribution by the brilliance and convolute if required

        if convolution is not None:
            df = self._convolute(df, convolution)
        Junits = self._get_brilliance(
            df, [x for x in norm_vars if x not in vars])

        if len(bins) != len(vars):
            print('List of variables and list of bins should have the same size')
            return

        xmed = [0]*len(bins)
        xleft = [0]*len(bins)
        xright = [0]*len(bins)
        dx = [0]*len(bins)

        for i, (bin, var, scale) in enumerate(zip(bins, vars, scales)):
            # If bins is int, create a mesh from var-min to var-max
            if type(bin) == int:
                if bin == 0:
                    if scale == 'log':
                        bins[i] = knuth_bin_width(
                            np.log10(df[var].to_numpy()), return_bins=True)[1]
                        bins[i] = 10**bins[i]
                    else:
                        bins[i] = knuth_bin_width(
                            df[var].to_numpy(), return_bins=True)[1]

                else:
                    if scale == 'log':
                        bins[i] = np.logspace(
                            np.log10(df[var].min()), np.log10(df[var].max()), bin)
                    else:
                        bins[i] = np.linspace(
                            df[var].min(), df[var].max(), bin)
            else:
                # If var is angle, convert degrees to radians
                if var == 'psi' or var == 'phi' or var == 'theta':
                    bins[i] = np.deg2rad(bins[i])

        # Calculate the histograms and the total current
        for i, bin in enumerate(bins):
            xleft[i] = bin[:-1]
            xright[i] = bin[1:]
            xmed[i] = 0.5*(xleft[i]+xright[i])
            dx[i] = xright[i]-xleft[i]

        xleft = np.meshgrid(*xleft, indexing='xy')
        xright = np.meshgrid(*xright, indexing='xy')
        xmed = np.meshgrid(*xmed, indexing='xy')
        dx = np.meshgrid(*dx, indexing='xy')

        p_mean = np.histogramdd(
            sample=df[vars].to_numpy(), bins=bins, weights=df['wgt'])[0]
        p_stdv = np.histogramdd(sample=df[vars].to_numpy(
        ), bins=bins, weights=df['wgt']**2)[0]**0.5
        p_total = ufloat(sum(df['wgt']), sum(df['wgt']**2)**0.5)

        p_mean, p_stdv = p_mean.T, p_stdv.T

        # Normalize the distribution considering the source factor, the number of source particles,
        # the brilliance, and the extra factor

        p_mean *= factor * self._brilliance * self._S0 / self._Np
        p_stdv *= factor * self._brilliance * self._S0 / self._Np
        p_total *= factor * self._brilliance * self._S0 / self._Np

        pinfo += self._get_info(df, norm_vars) + '\n'
        if type(factor) == int or type(factor) == float:
            pinfo += 'Integral: {:.3e}\n[{:s}]\n'.format(p_total, Junits)
        # TODO
        # if len(bins)==1:
            # pinfo += 'Max: {:.3e} at {:.3e} {:s}'.format(max(p_mean), xmed[0][p_mean==max(p_mean)], XUNITS[vars[0]])

        # Calculate the relative error
        with np.errstate(divide='ignore', invalid='ignore'):
            p_erel = np.nan_to_num(p_stdv / p_mean)

        for i, (bin, var) in enumerate(zip(bins, vars)):
            if var in norm_vars:
                p_mean = p_mean / YSCALE[var]
                p_stdv = p_stdv / YSCALE[var]
                if var == 'psi':
                    p_mean = p_mean / (-np.cos(xright[i]) + np.cos(xleft[i]))
                    p_stdv = p_stdv / (-np.cos(xright[i]) + np.cos(xleft[i]))
                elif var == 'R':
                    p_mean = p_mean / (xright[i]**2 / 2 - xleft[i]**2 / 2)
                    p_stdv = p_stdv / (xright[i]**2 / 2 - xleft[i]**2 / 2)
                elif var == 'lambda':
                    p_mean = p_mean / ((252.8 / 72.3)**-2 * (-xright[i]**-2 + xleft[i]**-2))
                    p_stdv = p_stdv / ((252.8 / 72.3)**-2 * (-xright[i]**-2 + xleft[i]**-2))
                else:
                    p_mean = p_mean / dx[i]
                    p_stdv = p_stdv / dx[i]

            # Convert the angles from radians to degrees
            if var == 'psi' or var == 'phi' or var == 'theta':
                bins[i] = np.rad2deg(bins[i])
                xleft[i] = np.rad2deg(xleft[i])
                xright[i] = np.rad2deg(xright[i])
                xmed[i] = np.rad2deg(xmed[i])

        # Save the results into a Dataframe
        p_df = pd.DataFrame()
        for i, (var, xl, xr, xm) in enumerate(zip(vars, xleft, xright, xmed)):
            p_df['{:s}-min'.format(var)] = xl.ravel()
            p_df['{:s}-max'.format(var)] = xr.ravel()
            p_df['{:s}'.format(var)] = xm.ravel()

        p_df['mean'] = p_mean.ravel()
        p_df['stdv'] = p_stdv.ravel()
        p_df['erel'] = p_erel.ravel()

        if total:
            return p_total, Junits
        else:
            return p_df, bins, pinfo

    def plot_distribution(self, vars, bins, scales=['linear', 'linear'], factor=1.0, filters={}, norm_vars=[], errors=True, convolution=None,
                          zlevels=5, info=False, ylabel=None, zlabel=None, zscale='linear', tolerance=1.0, **kwargs):
        """ Plot 1-D or 2-D distribution for given variables

        Parameters
        ----------
        var: str
            Variable to calculate the distribution
        bins: int or list
            Bins
            If int, then the distribution is going to be calculated between
            min(var), max(var) for
            Default: 100
        factor: float
            Multiply the distribution by a specific numerical factor
            Default: 1.0
        filters: dict
            Specify the other coordinates domain for the distribution
            It must be defined like: {'var':[var_min, var_max]}
            Default: {}
        **kwargs
            Extra arguments that will be passed to matplotlib (label, color, and so on)

        Returns
        -------

        """
        xscale, yscale = scales
        scales = [scales[i] for i in range(0, len(bins))]
        df, bins, pinfo = self.get_distribution(
            vars, bins, scales, factor, filters, norm_vars, total=False, convolution=convolution)
        if xscale == 'log':
            norm_vars = [var for var in norm_vars if var != vars[0]]
        if yscale == 'log' and len(vars) > 1:
            norm_vars = [var for var in norm_vars if var != vars[1]]
        Jlabel = self._get_units(norm_vars)

        if len(bins) == 1:
            Jbrill = 'Brilliance' if ylabel == None else ylabel
            Jlabel = '{:s}'.format(
                Jbrill) if Jlabel == '' else '{:s} [ {:s} ]'.format(Jbrill, Jlabel)
            if xscale == 'log':
                if info:
                    print("Plotting x*f(x) instead of f(x) (xscale='log')")
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[0])] * YSCALE[vars[0]]
                df['stdv'] = df['erel'] * df['mean']
            df.loc[df['stdv'] >= abs(df['mean']), 'stdv'] = 0.99999*df['stdv']
            df.loc[df['erel'] >= tolerance, 'mean'] = 0
            df.loc[df['erel'] >= tolerance, 'stdv'] = 0
            # df.loc[df['mean']==0, 'mean'] = np.nan
            if errors:
                plt.errorbar(df[vars], df['mean'], df['stdv'],
                             ds='steps-mid', **kwargs)
            else:
                plt.plot(df[vars].to_numpy(), df['mean'].to_numpy(),
                         ds='steps-mid', **kwargs)
            plt.xscale(xscale)
            plt.yscale(yscale)
            if XUNITS[vars[0]] != '':
                plt.xlabel(r'${:s}$ [{:s}]'.format(
                    XLATEX[vars[0]], XUNITS[vars[0]]))
            else:
                plt.xlabel(r'${:s}$'.format(XLATEX[vars[0]]))
            plt.ylabel(Jlabel)
        elif len(bins) == 2:
            Jbrill = 'Brilliance' if zlabel == None else zlabel
            Jlabel = '{:s}'.format(
                Jbrill) if Jlabel == '' else '{:s} [ {:s} ]'.format(Jbrill, Jlabel)

            len1 = len(bins[0])-1
            len2 = len(bins[1])-1
            bins1 = np.array(bins[0])
            bins2 = np.array(bins[1])

            if xscale == 'log' and yscale == 'log':
                if info:
                    print(
                        "Plotting x*y*f(x,y) instead of f(x,y) (xscale='log', yscale='log')")
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[0])] * YSCALE[vars[0]]
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[1])] * YSCALE[vars[1]]
                df['stdv'] = df['erel'] * df['mean']
            elif xscale == 'log':
                if info:
                    print("Plotting x*f(x,y) instead of f(x,y) (xscale='log')")
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[0])] * YSCALE[vars[0]]
                df['stdv'] = df['erel'] * df['mean']

            elif yscale == 'log':
                if info:
                    print("Plotting y*f(x,y) instead of f(x,y) (xscale='log')")
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[1])] * YSCALE[vars[1]]
                df['stdv'] = df['erel'] * df['mean']

            df.loc[df['erel'] >= tolerance, 'mean'] = 0
            df.loc[df['erel'] >= tolerance, 'stdv'] = 0
            df.loc[df['mean'] == 0, 'mean'] = np.nan

            z_mean = df['mean'].to_numpy().reshape(len2, len1)
            # z_stdv = df['stdv'].to_numpy().reshape(len2, len1)
            if zscale == 'linear':
                plt.pcolor(0.5*(bins1[1:]+bins1[:-1]), 0.5*(bins2[1:] +
                           bins2[:-1]), z_mean, shading='auto', **kwargs)
                cbar = plt.colorbar(label=Jlabel)
                lvls = np.linspace(cbar.vmin, cbar.vmax, zlevels)
            elif zscale == 'log':
                plt.pcolor(0.5*(bins1[1:]+bins1[:-1]), 0.5*(bins2[1:]+bins2[:-1]),
                           z_mean, shading='auto', norm=LogNorm(), **kwargs)
                cbar = plt.colorbar(label=Jlabel)
                lvls = np.logspace(np.floor(np.log10(cbar.vmin)),
                                   np.ceil(np.log10(cbar.vmax)),
                                   int(np.ceil(np.log10(cbar.vmax))-np.floor(np.log10(cbar.vmin))+1))

            plt.xscale(xscale)
            plt.yscale(yscale)
            if XUNITS[vars[0]] != '':
                plt.xlabel(r'${:s}$ [{:s}]'.format(
                    XLATEX[vars[0]], XUNITS[vars[0]]))
            else:
                plt.xlabel(r'${:s}$'.format(XLATEX[vars[0]]))
            if XUNITS[vars[1]] != '':
                plt.ylabel(r'${:s}$ [{:s}]'.format(
                    XLATEX[vars[1]], XUNITS[vars[1]]))
            else:
                plt.ylabel(r'${:s}$'.format(XLATEX[vars[1]]))
            cntrs = plt.contour(0.5*(bins[0][1:]+bins[0][:-1]), 0.5*(bins[1][1:]+bins[1][:-1]),
                                z_mean, colors='grey', levels=lvls, vmin=cbar.vmin, vmax=cbar.vmax)
            cbar.add_lines(cntrs)

        if info:
            plt.figtext(x=0.92, y=0.1, s=r'{:s}'.format(
                pinfo), fontdict={'size': 8})

    def save_source_file(self, filepath, **kwargs):
        new_extension = os.path.splitext(filepath)[-1]
        df = self._df2.copy()  # get_pandas_dataframe()
        df = df[MCPLColumns]

        # Write in the OpenMC .h5 format
        if new_extension == '.h5':
            for pin, pout in zip((PDGCode), (OpenMCCode)):
                df.loc[df['type'] == pin.value, 'type'] = pout.value

            pos_dtype = np.dtype([('x', '<f8'), ('y', '<f8'), ('z', '<f8')])
            source_dtype = np.dtype([('r', pos_dtype),
                                     ('u', pos_dtype),
                                     ('E', '<f8'),
                                     ('time', '<f8'),
                                     ('wgt', '<f8'),
                                     ('delayed_group', '<i4'),
                                     ('surf_id', '<i4'),
                                     ('particle', '<i4'), ])
            arr = np.array([((s[3], s[4], s[5]),
                             (s[6], s[7], s[8]),
                             (s[2]*1e6), (s[9]*1e-3),
                             (s[10]), (0), (0), (s[1])) for s in df.values], dtype=source_dtype)

            kwargs.setdefault('mode', 'w')
            with h5py.File(filepath, **kwargs) as fh:
                fh.attrs['filetype'] = np.string_("source")
                fh.create_dataset('source_bank', data=arr, dtype=source_dtype)

        # Write in MCPL format
        elif new_extension == '.mcpl':
            self.save_source_file('temp.txt')
            subprocess.call(['ascii2mcpl', 'temp.txt', filepath])
            subprocess.call(['rm', 'temp.txt'])

        # Write the ASCII-based format file in the other cases
        else:
            with open(filepath, 'w') as fo:
                fo.write('#MCPL-ASCII\n')
                fo.write('#GENERATED FROM MCTAL-MCPL\n')
                fo.write('#NPARTICLES: {:d}\n'.format(len(df)))
                fo.write('#END-HEADER\n')
                fo.write("index     pdgcode               ekin[MeV]                   x[cm]          "
                         + "         y[cm]                   z[cm]                      ux                  "
                         + "    uy                      uz                time[ms]                  weight  "
                         + "                 pol-x                   pol-y                   pol-z  userflags\n")

                fmtstr = "%5i %11i %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g %23.18g 0x%08x\n"
                for i, s in enumerate(df.values):
                    # fo.write('{:.0f}\t{:.0f}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:s}\n'.format(*[j for j in s]))
                    fo.write(fmtstr % (s[0], s[1], s[2], s[3], s[4], s[5], s[6],
                             s[7], s[8], s[9], s[10], s[11], s[12], s[13], int(s[14])))


if __name__ == "__main__":
    SurfaceSourceFile()
