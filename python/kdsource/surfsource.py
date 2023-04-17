import os
import subprocess
from enum import Enum
from math import cos, pi, sin

from astropy.stats import knuth_bin_width
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mcpl
import numpy as np
import pandas as pd
from uncertainties import ufloat

# plt.style.use(os.path.dirname(os.path.abspath(__file__)) + '/plots.mplstyle')
plt.style.use('/home/nschmidt/kdsource/develop/python/kdsource/plots.mplstyle')

c2 = 8.98755e16        # m2 / s2
hc = 1.23984186e-12    # MeV m


MCPLColumns = ['id', 'type', 'E', 'x', 'y', 'z', 'u', 'v', 'w',
               't', 'wgt', 'px', 'py', 'pz', 'userflags']
XUNITS = {'x': 'cm', 'y': 'cm', 'z': 'cm', 'R': 'cm', 'theta': '$^o$',
          'u': '', 'v': '', 'w': '', 'phi': '$^o$', 'psi': '$^o$', 'mu': '',
          'E': 'MeV', 'ln(E0/E)': '', 'lambda': r'$\AA$',
          't': 'ms', 'log(t)': 'ms'}
YUNITS = {'x': 'cm$^{-1}$', 'y': 'cm$^{-1}$', 'z': 'cm$^{-1}$',
          'R': 'cm$^{-2}$', 'theta': 'rad$^{-1}$', 'u': '', 'v': '', 'w': '',
          'phi': 'rad$^{-1}$', 'psi': 'rad$^{-1}$', 'mu': '',
          'E': 'eV$^{-1}$', 'ln(E0/E)': '', 'lambda': r'$\AA^{-1}$',
          't': 's$^{-1}$', 'log(t)': ''}
XLATEX = {'x': 'x', 'y': 'y', 'z': 'z', 'R': 'R', 'theta': r'\theta',
          'u': 'u', 'v': 'v', 'w': 'w', 'phi': r'\varphi',
          'psi': r'\vartheta', 'mu': r'\mu=\cos(\vartheta)',
          'E': 'E', 'lambda': r'\lambda', 'ln(E0/E)': r'\ln(E_0/E)',
          't': 't', 'log(t)': r'\log(t)'}
YSCALE = {'x': 1, 'y': 1, 'z': 1, 'R': 1, 'theta': 1,
          'u': 1, 'v': 1, 'w': 1, 'phi': 1, 'psi': 1, 'mu': 1,
          'E': 1e6, 'ln(E0/E)': 1, 'lambda': 1,
          't': 1e-3, 'log(t)': 1}
PARTMASS = {2112: 939.5653, 22: 0, 11: 0.511, 2212: 938.7829}

def momentum(ptype, pekin):
    mc2 = np.array([PARTMASS[np.abs(i)] for i in ptype])
    pc = pekin * np.sqrt(1. + 2 * mc2 / pekin) 
    return pc

def velocity(ptype, pekin):
    mc2 = np.array([PARTMASS[np.abs(i)] for i in ptype])
    pc = momentum(ptype, pekin)
    v = np.array([pci / mc2i if mc2i > 0 else c2 ** 0.5 for pci, mc2i in zip(pc, mc2)])
    return v


def wavelength(ptype, pekin):
    return hc / momentum(ptype, pekin) * 1e10



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


class VitessCode(Enum):
    NEUTRON = 'N'
    PHOTON = 'G'
    ELECTRON = 'E-'
    POSITRON = 'E+'
    PROTON = 'P'


class SurfaceSourceFile:
    """ Read a surface source file.
    Possible formats: MCPL, HDF5 (OpenMC), SSV (KDSource)

    Parameters
    ----------
    filepath: str
        Path to surface source file.
    S0: float
        Simulation source factor.
        Default: 1.0
    Nparticles: int
        Total source particles simulated.
        Default: 1e9
    dA: float
        Area of the surface [cm$^2$].
        Default: None
    translation: list
        Translation for the position variables [cm].
        Default: [0, 0, 0]
    rotation: list
        Rotation for the position and direction variables [deg].
        Default: [0, 0, 0]
    domain: dict
        Range of the variables: var_min <= 'var' < var_max.
        It must be defined like: {'var':[var_min, var_max]}.
        List of possible variables:
        type [PDG],
        E [MeV], ln(E0/E), lambda [AA],
        x [cm], y [cm], z [cm], R [cm], theta [rad],
        u, v, w, phi [rad], psi [rad],
        t [ms], log(t),
        wgt.
        Default: {}
    set_domain_first: bool
        Define if the variables domains and the normal direction
        must be set before or after the translation and rotation.
        Default: False
    set_rotation_first: bool
        Define if the rotation must be done before or after
        the translation of the coordinates.
        Default: False
    E0: float
        Reference energy [MeV] to calculate the lethargy ln(E0/E).
        Default: 20
    convoluted: bool
        Define if the simulation was performed with a convoluted pulse or not.
        Default: False
    uvw_reference: str
        Versor which is normal to the surface source after the rotation.
        Default: 'w'
    surface: int
        Define the surface id to extract the particles from the tracks file.
        Default: None
    Nmax: int
        Define the maximum number of particles to be readed.
    current: float
        Current, in mA, to normalize the results.
    tpulse: float
        Pulse width, in s, to normalize the results.
    """

    def __init__(self, filepath, S0=1.0, Nparticles=1e9, dA=None,
                 translation=[0, 0, 0], rotation=[0, 0, 0], domain={},
                 set_domain_first=False, set_rotation_first=False, E0=20,
                 convoluted=False, uvw_reference='w', surface=None, Nmax=1e9,
                 current=None, tpulse=None, skip_cloned=[]):
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
        self._surface = surface
        self._max_particles = int(Nmax)
        self._tpulse = tpulse
        self._current = current
        self._convoluted = convoluted
        self._cloned = skip_cloned
        self._df = self.__read__()
        self._df2 = self.get_pandas_dataframe()
        self._brilliance = 1.0

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
                df['E'] = fh['source_bank']['E'] * 1e-6  # eV to MeV
                df['x'] = fh['source_bank']['r']['x']  # cm
                df['y'] = fh['source_bank']['r']['y']  # cm
                df['z'] = fh['source_bank']['r']['z']  # cm
                df['u'] = fh['source_bank']['u']['x']
                df['v'] = fh['source_bank']['u']['y']
                df['w'] = fh['source_bank']['u']['z']
                df['wgt'] = fh['source_bank']['wgt']
                df['t'] = fh['source_bank']['time'] * 1e3  # s to ms
                df['px'] = 0.0
                df['py'] = 0.0
                df['pz'] = 0.0
                df['userflags'] = 0x00000000
                df = df[df['id'] < self._max_particles]
        # MCPL format
        elif self._extension == '.gz' or self._extension == '.mcpl':
            with mcpl.MCPLFile(self._filepath,
                               blocklength=self._max_particles) as mcpl_file:
                pb = mcpl_file.read_block()
                plist = np.array([pb.pdgcode, pb.ekin, pb.x, pb.y, pb.z,
                                  pb.ux, pb.uy, pb.uz, pb.time, pb.weight,
                                  pb.polx, pb.poly, pb.polz, pb.userflags])
                df = pd.DataFrame(plist.T, columns=MCPLColumns[1:])
                if self._surface is not None:
                    df = df[df['userflags'] == int(self._surface)]
                df['id'] = np.arange(0, len(df))
                df = df[MCPLColumns]
        # VITESS format (for detectors, not for dump files)
        elif self._extension == '.vitess':
            df = pd.DataFrame(np.loadtxt(self._filepath,
                                         usecols=list(np.arange(3, 15))),
                              columns=['t', 'E', 'wgt', 'z', 'x', 'y',
                                       'w', 'u', 'v', 'px', 'py', 'pz'])
            df['E'] = (72.3 / (252.8 * df['E'].to_numpy())) ** 2 * 1e-6
            df['id'] = [int(i[2:])-1 for i in np.loadtxt(self._filepath,
                                                       dtype='str',
                                                       usecols=(0))]
            df['type'] = np.loadtxt(self._filepath, dtype='str', usecols=(1))
            df['userflags'] = 0x00000000
            for pin, pout in zip((VitessCode), (PDGCode)):
                df.loc[df['type'] == pin.value, 'type'] = pout.value
            df = df[MCPLColumns]
        # ASCII format, for all the other cases
        else:
            df = pd.DataFrame(np.loadtxt(
                self._filepath, skiprows=5), columns=MCPLColumns)

        if len(self._cloned) > 0:
            df = self.search_duplicated_particles(df, self._cloned)

        # Print some info regarding the particles in the file
        print('Number of particles in file {:s}: {:d}'.format(
            self._filepath, len(df)))
        for p, idx in zip((PDGCode), ['n', 'g', 'e-', 'e+', 'p']):
            print('{:.0f}% {:s}, '.format(
                len(df[df['type'] == p.value]) / len(df['type']) * 100, idx),
                end='')
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
        if self._tpulse is not None and self._convoluted is False:
            self._convolute(df, {'t': self._tpulse * 1e3})  # from s to ms
            self._convoluted = True
        # Calculate extra coordinates
        df['R'] = (df[X].to_numpy()**2 + df[Y].to_numpy()**2)**0.5
        df['theta'] = (np.arctan2(df[Y].to_numpy(), df[X].to_numpy()))
        df['mu'] = df[W].to_numpy()
        df['psi'] = (np.arccos(df[W].to_numpy()))
        df['phi'] = (np.arctan2(df[V].to_numpy(), df[U].to_numpy()))
        df['ln(E0/E)'] = np.log(self._E0 / df['E'].to_numpy())
        df['log(t)'] = np.log10(df['t'].to_numpy())
        # df['lambda'] = 72.3 / 252.8 * (df['E'] * 1e6)**-0.5
        df['lambda'] = wavelength(df['type'].to_numpy(), df['E'].to_numpy())
        # Check domain
        if not self._domain_first:
            df = self._set_domain(df)
        return df

    def search_duplicated_particles(self, df, vars):
        newwgt = df.groupby(vars, sort = False, as_index = False,
                            ).sum(numeric_only = True)['wgt'].values
        newdf = df.drop_duplicates(vars,
                                   ignore_index = True).copy()
        newdf['wgt'] = newwgt
        return newdf

    def _set_translation(self, df):
        # Translate the coordinates (only position)
        df['x'] = df['x'].to_numpy() + self._translation[0]
        df['y'] = df['y'].to_numpy() + self._translation[1]
        df['z'] = df['z'].to_numpy() + self._translation[2]
        return df

    def _set_rotation(self, df):
        # Rotate the coordinates (posisition and direction)
        phi, theta, psi = np.array(self._rotation) * (pi / 180)
        c3, s3 = cos(phi), sin(phi)
        c2, s2 = cos(theta), sin(theta)
        c1, s1 = cos(psi), sin(psi)
        rotation_matrix = np.array([
            [c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2],
            [c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3],
            [-s2, c2 * s3, c2 * c3]])
        df['x'], df['y'], df['z'] = rotation_matrix @ np.array(
            [df['x'].to_numpy(), df['y'].to_numpy(), df['z'].to_numpy()])
        df['u'], df['v'], df['w'] = rotation_matrix @ np.array(
            [df['u'].to_numpy(), df['v'].to_numpy(), df['w'].to_numpy()])
        return df

    def _normalize_direction(self, df):
        # Normalize the direction versor
        unorm = df['u'].to_numpy()**2 + df['v'].to_numpy()**2 + \
            df['w'].to_numpy()**2

        df['u'], df['v'], df['w'] = (df['u'].to_numpy() / unorm**0.5,
                                     df['v'].to_numpy() / unorm**0.5,
                                     df['w'].to_numpy() / unorm**0.5)
        return df

    def _set_domain(self, df):
        # Check the filters coordinates
        for pvar, (pmin, pmax) in self._domain.items():
            if pvar == 'psi' or pvar == 'phi' or pvar == 'theta':
                pmin, pmax = np.deg2rad([pmin, pmax])
            if pmin is not None:
                df = df[df[pvar] >= pmin]
            if pmax is not None:
                df = df[df[pvar] < pmax]
        return df

    def _convolute(self, df, convolution):
        # Convolute the desired variables
        # Set the seed to reproduce the results
        np.random.seed(len(df))
        for cvar, cwidth in convolution.items():
            df[cvar] = df[cvar].to_numpy() \
                + np.random.uniform(low=0.0, high=cwidth, size=len(df[cvar]))
        return df

    def _get_brilliance(self, df, norm_vars):
        self._brilliance = 1.0
        Sunit = self._get_units(norm_vars)

        if self._dA is not None:
            self._brilliance /= self._dA

        if 'mAs' in norm_vars:
            self._brilliance /= (1.6022e-19 * 1e3)
            if self._current is not None:
                self._brilliance *= self._current
            if self._tpulse is not None:
                self._brilliance *= self._tpulse
            norm_vars = [x for x in norm_vars if x not in ['mAs']]

        for nvar in norm_vars:
            pmin = min(df[nvar]) * YSCALE[nvar]
            pmax = max(df[nvar]) * YSCALE[nvar]
            if nvar == 'R':
                dp = 0.5 * (pmax**2 - pmin**2)
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
            if self._current is None and self._tpulse is None:
                Sunit += '(mA s)$^{-1}$ '
            elif self._current is None:
                Sunit += 'mA$^{-1}$ '
            elif self._tpulse is None:
                Sunit += 's$^{-1}$ '
            else:
                if 't' not in norm_vars:
                    Sunit += 'pulse$^{-1}$ '
            norm_vars = [x for x in norm_vars if x not in ['mAs']]

        if self._dA is not None:
            Sunit += 'cm$^{-2}$ '
            dA = True

        if 'x' in norm_vars and 'y' in norm_vars and dA is False:
            Sunit += 'cm$^{-2}$ '
            norm_vars = [x for x in norm_vars if x not in ['x', 'y']]

        if 'x' in norm_vars and 'z' in norm_vars and dA is False:
            Sunit += 'cm$^{-2}$ '
            norm_vars = [x for x in norm_vars if x not in ['x', 'z']]

        if 'y' in norm_vars and 'z' in norm_vars and dA is False:
            Sunit += 'cm$^{-2}$ '
            norm_vars = [x for x in norm_vars if x not in ['y', 'z']]

        if 'R' in norm_vars and 'theta' in norm_vars and dA is False:
            Sunit += 'cm$^{-2}$ '
            norm_vars = [x for x in norm_vars if x not in ['R', 'theta']]

        if 'mu' in norm_vars and 'phi' in norm_vars and dOmega is False:
            Sunit += 'sr$^{-1}$ '
            norm_vars = [x for x in norm_vars if x not in ['mu', 'phi']]
            dOmega = True

        if 'psi' in norm_vars and 'phi' in norm_vars and dOmega is False:
            Sunit += 'sr$^{-1}$ '
            norm_vars = [x for x in norm_vars if x not in ['psi', 'phi']]
            dOmega = True

        for nvar in norm_vars:
            Sunit += '{:s} '.format(YUNITS[nvar])

        return Sunit

    def _get_info(self, df, vars):
        info = 'Surface: {:d}\n\n'.format(self._surface if self._surface is not None else 0)
        info += 'NSIM: {:.2e}\n\n'.format(self._Np)
        info += 'NMCPL: {:d}\n\n'.format(len(df))
        info += 'S0: {:.2e}\n\n'.format(self._S0)
        info += 'Area: {:.2e} cm$^2$\n\n'.format(self._dA if self._dA is not None else 0.0)
        info += 'Translation: [{:.1f}, {:.1f}, {:.1f}]\n\n'.format(
            self._translation[0], self._translation[1], self._translation[2])
        info += 'Rotation: [{:.1f}, {:.1f}, {:.1f}]\n\n'.format(
            self._rotation[0], self._rotation[1], self._rotation[2])
        for pvar in vars:
            if pvar != 'mAs':
                pmin = min(df[pvar])
                pmax = max(df[pvar])
                if pvar == 'theta' or pvar == 'psi' or pvar == 'phi':
                    pmin, pmax = np.rad2deg([pmin, pmax])
                info += '${:s}$: [{:.2e}, {:.2e}] {:s}'.format(
                    XLATEX[pvar], pmin, pmax, XUNITS[pvar]) + '\n\n'
            else:
                if self._current is not None:
                    info += 'Current: {:.2f} mA\n\n'.format(self._current)
                if self._tpulse is not None:
                    info += 'Pulse width: {:.2f} us\n\n'.format(
                        self._tpulse * 1e6)
        return info

    def get_distribution(self, vars, bins, scales=['linear'], factor=1.0,
                         filters={}, norm_vars=[], convolution=None,
                         total=False,):
        """ Obtain a N-D distribution for given variables

        Parameters
        ----------
        vars: list of str
            Variables to calculate the distribution
        bins: list of int or list of iterable
            Number or array of bins to calculate the distribution.
            If int, then the distributions are going to be calculated between
            min(var), max(var) for int bins.
            If 0, the number of bins will be calculated optimally with
            the knuth_bin_width function from astropy.
        scales: list of str
            Scales of the axis. The options are linear or log.
            Default: ['linear']
        factor: float
            Multiply the distribution by a specific numerical factor.
            Default: 1.0
        filters: dict
            Specify the other coordinates domain for the distribution
            It must be defined like: {'var':[var_min, var_max]}
            Default: {}
        norm_vars: list of str
            List of the variables to calculate the brilliance, i.e. to
            normalize the distribution.
            List of possible variables:
            mAs [mAs], E [MeV], ln(E0/E), lambda [AA],
            x [cm], y [cm], z [cm], R [cm], theta [rad],
            u, v, w, phi [rad], psi [rad],
            t [ms], log(t),
        convolution: dict
            Specify the coordinates to perform a convolution.
            It must be defined like: {'var':[var_convolution]}
            Default: {}
        total: bool
            Define if the integral value is required.
            Default: False

        Returns
        -------
        p_df: DataFrame
            Pandas DataFrame with the bins and the distribution.
        bins: list of arrays
            List of the bins where the distributions were calculated.
        p_info: str
            String with the information of the distributions calculated.
        p_total: ufloat
            If total is True, the integral value of the distribution for the
            required normalization variables.
        p_units: str
            If total is True, the units for the integral value of the
            distribution.
        """
        p_info = ''
        old_filters = self._domain
        self._domain = filters
        # df = self.get_pandas_dataframe()
        df = self._df2.copy()
        df = self._set_domain(df)
        self._domain = old_filters

        # Normalize the distribution by the brilliance and convolute
        if convolution is not None:
            df = self._convolute(df, convolution)
        p_units = self._get_brilliance(
            df, [x for x in norm_vars if x not in vars])

        if len(bins) != len(vars):
            print('Both lists of variables and bins should have the same size')
            return

        xmed = [0] * len(bins)
        xleft = [0] * len(bins)
        xright = [0] * len(bins)
        dx = [0] * len(bins)

        for i, (bin, var, scale) in enumerate(zip(bins, vars, scales)):
            # If bins is int, create a mesh from var-min to var-max
            if type(bin) == int:
                if bin == 0:
                    if scale == 'log':
                        bins[i] = knuth_bin_width(np.log10(df[var].to_numpy()),
                                                  return_bins=True)[1]
                        bins[i] = 10**bins[i]
                    else:
                        bins[i] = knuth_bin_width(df[var].to_numpy(),
                                                  return_bins=True)[1]

                else:
                    if scale == 'log':
                        bins[i] = np.logspace(np.log10(df[var].min()),
                                              np.log10(df[var].max()),
                                              bin)
                    else:
                        bins[i] = np.linspace(df[var].min(),
                                              df[var].max(),
                                              bin)
            else:
                # If var is angle, convert degrees to radians
                if var == 'psi' or var == 'phi' or var == 'theta':
                    bins[i] = np.deg2rad(bins[i])

        # Calculate the histograms and the total current
        for i, bin in enumerate(bins):
            xleft[i] = bin[:-1]
            xright[i] = bin[1:]
            xmed[i] = 0.5 * (xleft[i] + xright[i])
            dx[i] = xright[i] - xleft[i]

        xleft = np.meshgrid(*xleft, indexing='ij')
        xright = np.meshgrid(*xright, indexing='ij')
        xmed = np.meshgrid(*xmed, indexing='ij')
        dx = np.meshgrid(*dx, indexing='ij')

        p_mean = np.histogramdd(
            sample=df[vars].to_numpy(), bins=bins, weights=df['wgt'])[0]
        p_stdv = np.histogramdd(sample=df[vars].to_numpy(
        ), bins=bins, weights=df['wgt']**2)[0]**0.5
        p_total = ufloat(sum(df['wgt']), sum(df['wgt']**2)**0.5)

        # p_mean, p_stdv = p_mean.T, p_stdv.T

        # Normalize the distribution considering the source factor,
        # the number of source particles, the brilliance, and the extra factor
        p_mean *= factor * self._brilliance * self._S0 / self._Np
        p_stdv *= factor * self._brilliance * self._S0 / self._Np
        p_total *= factor * self._brilliance * self._S0 / self._Np

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
                else:
                    p_mean = p_mean / dx[i]
                    p_stdv = p_stdv / dx[i]

            # Convert the angles from radians to degrees
            if var == 'psi' or var == 'phi' or var == 'theta':
                bins[i] = np.rad2deg(bins[i])
                xleft[i] = np.rad2deg(xleft[i])
                xright[i] = np.rad2deg(xright[i])
                xmed[i] = np.rad2deg(xmed[i])

        p_info += self._get_info(df, norm_vars)
        if len(bins)==1:
            p_info += 'Max: {:.3e} at {:.3e} {:s}\n\n'.format(
                max(p_mean), xmed[0][p_mean==max(p_mean)][0], XUNITS[vars[0]])
        if type(factor) == int or type(factor) == float:
            p_info += 'Integral: {:.3e} [{:s}]\n'.format(p_total, p_units)        
        
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
            return p_total, p_units
        else:
            return p_df, bins, p_info
        
    def get_domain(self, vars, filters={}, ):        
        old_filters = self._domain
        self._domain = filters
        df = self._df2.copy()
        df = self._set_domain(df)
        self._domain = old_filters

        dm = {}
        for var in vars:
            dm[var] = df[var].min(), df[var].max()
            if var == 'theta' or var == 'psi' or var == 'phi':
                dm[var] = np.rad2deg([dm[var][0], dm[var][1]])      
        return dm


    def plot_distribution(self, vars, bins, scales=['linear', 'linear'],
                          factor=1.0, filters={}, norm_vars=[], errors=True,
                          convolution=None, zlevels=1, info=False, ylabel=None,
                          zlabel=None, zscale='linear', tolerance=1.0,
                          vmin = None, vmax = None, peak_brilliance = False,
                          **kwargs):
        """ Plot 1-D or 2-D distribution for given variables

        Parameters
        ----------
        vars: list of str
            Variables to calculate the distribution
        bins: list of int or list of iterable
            Number or array of bins to calculate the distribution.
            If int, then the distributions are going to be calculated between
            min(var), max(var) for int bins.
            If 0, the number of bins will be calculated optimally with
            the knuth_bin_width function from astropy.
        scales: list of str
            Scales of the axis. The options are linear or log.
            Default: ['linear', 'linear']
        factor: float
            Multiply the distribution by a specific numerical factor.
            Default: 1.0
        filters: dict
            Specify the other coordinates domain for the distribution
            It must be defined like: {'var':[var_min, var_max]}
            Default: {}
        norm_vars: list of str
            List of the variables to calculate the brilliance, i.e. to
            normalize the distribution.
        errors: bool
            Specify if the error bars should be plotted or not.
            Default: True
        convolution: dict
            Specify the coordinates to perform a convolution.
            It must be defined like: {'var':[var_convolution]}
            Default: {}
        zlevels: int
            Number of countour surfaces to plot with the colormap (if more
            than one variable is required). Deprecated when zscale is 'log'
            Default: 1
        info: bool
            Specify if the information of the plot should be included or not.
            Default: False
        ylabel: str
            Label for the y-axis.
            Default: 'Intensity' (for 1-D plots)
        zlabel: str
            Label for the z-axis (2-D plots).
            Default: 'Intensity'
        zscale: str
            Scale for the colorbar (2-D plots).
            Default: 'linear'
        tolerance: float
            When the realive error is greater or equal than the tolerance,
            the mean values of the distribution are not shown.
            Default: 1.0
        **kwargs
            Extra arguments that will be passed to matplotlib
            (label, color, and so on)

        Returns
        -------
        matplotlib 1-D or 2-D plot for the required variables with the proper
        normalization.
        """
        xscale, yscale = scales
        scales = [scales[i] for i in range(0, len(bins))]
        df, bins, pinfo = self.get_distribution(
            vars, bins, scales, factor, filters, norm_vars, convolution)
        if peak_brilliance and len(vars) == 2:
            df = df.groupby([var for var in vars if var != 't']).max().reset_index()
        if xscale == 'log':
            norm_vars = [var for var in norm_vars if var != vars[0]]
        if yscale == 'log' and len(vars) > 1:
            norm_vars = [var for var in norm_vars if var != vars[1]]
        Jlabel = self._get_units(norm_vars)

        if len(bins) == 1:
            Jbrill = 'Intensity' if ylabel is None else '{:s}'.format(ylabel)
            Jlabel = '{:s}'.format(Jbrill) if Jlabel == '' \
                else '{:s} [ {:s} ]'.format(Jbrill, Jlabel)
            if xscale == 'log':
                if info:
                    print("Plotting x*f(x) instead of f(x) (xscale='log')")
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[0])] * YSCALE[vars[0]]
                df['stdv'] = df['erel'] * df['mean']
            df.loc[df['erel'] >= tolerance, 'mean'] = 0
            df.loc[df['erel'] >= tolerance, 'stdv'] = 0
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
            plt.ylabel(r'{:s}'.format(Jlabel))

        elif len(bins) == 2:
            Jbrill = 'Intensity' if zlabel is None else zlabel
            Jlabel = '{:s}'.format(Jbrill) if Jlabel == ''\
                else '{:s} [ {:s} ]'.format(Jbrill, Jlabel)

            len1 = len(bins[0]) - 1
            len2 = len(bins[1]) - 1
            bins1 = np.array(bins[0])
            bins2 = np.array(bins[1])

            if xscale == 'log' and yscale == 'log':
                if info:
                    print("Plotting x*y*f(x,y) instead of f(x,y)\
                        (xscale='log', yscale='log')")
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[0])] * YSCALE[vars[0]]
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[1])] * YSCALE[vars[1]]
                df['stdv'] = df['erel'] * df['mean']
            elif xscale == 'log':
                if info:
                    print("Plotting x*f(x,y) instead of f(x,y)\
                        (xscale='log')")
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[0])] * YSCALE[vars[0]]
                df['stdv'] = df['erel'] * df['mean']

            elif yscale == 'log':
                if info:
                    print("Plotting y*f(x,y) instead of f(x,y)\
                        (yscale='log')")
                df['mean'] = df['mean'] * \
                    df['{:s}'.format(vars[1])] * YSCALE[vars[1]]
                df['stdv'] = df['erel'] * df['mean']

            df.loc[df['erel'] >= tolerance, 'mean'] = 0
            df.loc[df['erel'] >= tolerance, 'stdv'] = 0
            df.loc[df['mean'] == 0, 'mean'] = np.nan

            z_mean = df['mean'].to_numpy().reshape(len1, len2).T
            # z_stdv = df['stdv'].to_numpy().reshape(len2, len1)
            if zscale == 'linear':
                plt.pcolor(0.5 * (bins1[1:] + bins1[:-1]),
                           0.5 * (bins2[1:] + bins2[:-1]),
                           z_mean, shading='auto', **kwargs)
                cbar = plt.colorbar(label=r'{:s}'.format(Jlabel))
                if zlevels > 0:
                    lvls = np.linspace(cbar.vmin, cbar.vmax, zlevels)
            elif zscale == 'log':
                plt.pcolor(0.5 * (bins1[1:] + bins1[:-1]),
                           0.5 * (bins2[1:] + bins2[:-1]),
                           z_mean, shading='auto', norm=LogNorm(vmin, vmax), **kwargs)
                cbar = plt.colorbar(label=r'{:s}'.format(Jlabel))
                if zlevels > 0:
                    lvls = np.logspace(
                        np.floor(np.log10(cbar.vmin)),
                        np.ceil(np.log10(cbar.vmax)),
                        zlevels*int(np.ceil(np.log10(cbar.vmax))
                            - np.floor(np.log10(cbar.vmin))) + 1)

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

            if zlevels > 0:
                cntrs = plt.contour(0.5 * (bins[0][1:] + bins[0][:-1]),
                                    0.5 * (bins[1][1:] + bins[1][:-1]),
                                    z_mean, colors='grey', levels=lvls,
                                    vmin=cbar.vmin, vmax=cbar.vmax)
                cbar.add_lines(cntrs)

        if info:
            plt.figtext(x=0.92, y=0.1, s=r'{:s}'.format(
                pinfo), fontdict={'size': 8})

    def save_source_file(self, filepath, **kwargs):
        create_source_file(self._df2.copy(), filepath, **kwargs)

def create_source_file(df, filepath, **kwargs):
    new_extension = os.path.splitext(filepath)[-1]
    df = df[MCPLColumns]

    # Write in the OpenMC .h5 format
    if new_extension == '.h5':
        print('Saving into OpenMC format (HDF5)')
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
        arr = np.array([(
            (s[3], s[4], s[5]),
            (s[6], s[7], s[8]),
            (s[2] * 1e6), (s[9] * 1e-3),
            (s[10]), (0), (0), (s[1]))
            for s in df.values], dtype=source_dtype)
        kwargs.setdefault('mode', 'w')
        with h5py.File(filepath, **kwargs) as fh:
            fh.attrs['filetype'] = np.string_("source")
            fh.create_dataset('source_bank', data=arr, dtype=source_dtype)

    # Write in MCPL format
    elif new_extension == '.mcpl' or new_extension == '.gz':
        new_extension = '.mcpl'
        print('Saving into MCPL format')
        create_source_file(df, 'temp.txt')
        subprocess.call(['ssv2mcpl', 'temp.txt', filepath])
        subprocess.call(['rm', 'temp.txt'])

    # Write the ASCII-based format file in the other cases
    else:
        print('Saving into SSV format (ASCII)')
        with open(filepath, 'w') as fo:
            sheader = ''
            fmtstr = ''

            sheader += '#MCPL-ASCII\n'
            sheader += '#GENERATED FROM KDSOURCE\n'
            sheader += '#NPARTICLES: {:d}\n'.format(len(df))
            sheader += '#END-HEADER\n'
            sheader += "index     "
            sheader += "pdgcode               "
            sheader += "ekin[MeV]                   "
            sheader += "x[cm]                   "
            sheader += "y[cm]                   "
            sheader += "z[cm]                      "
            sheader += "ux                      "
            sheader += "uy                      "
            sheader += "uz                "
            sheader += "time[ms]                  "
            sheader += "weight                   "
            sheader += "pol-x                   "
            sheader += "pol-y                   "
            sheader += "pol-z  "
            sheader += "userflags\n"
            fo.write(sheader)

            fmtstr += "%5i %11i %23.18g %23.18g %23.18g %23.18g "
            fmtstr += "%23.18g %23.18g %23.18g %23.18g %23.18g "
            fmtstr += "%23.18g %23.18g %23.18g 0x%08x\n"
            for s in df.values:
                fo.write(fmtstr % (s[0], s[1], s[2], s[3], s[4], s[5],
                                    s[6], s[7], s[8], s[9], s[10], s[11],
                                    s[12], s[13], int(s[14])))
    print('Done, saved into {:s} file'.format(filepath))



import matplotlib as mpl
from matplotlib.colors import ListedColormap
mako = [
    [ 0.04503935, 0.01482344, 0.02092227],
    [ 0.04933018, 0.01709292, 0.02535719],
    [ 0.05356262, 0.01950702, 0.03018802],
    [ 0.05774337, 0.02205989, 0.03545515],
    [ 0.06188095, 0.02474764, 0.04115287],
    [ 0.06598247, 0.0275665 , 0.04691409],
    [ 0.07005374, 0.03051278, 0.05264306],
    [ 0.07409947, 0.03358324, 0.05834631],
    [ 0.07812339, 0.03677446, 0.06403249],
    [ 0.08212852, 0.0400833 , 0.06970862],
    [ 0.08611731, 0.04339148, 0.07538208],
    [ 0.09009161, 0.04664706, 0.08105568],
    [ 0.09405308, 0.04985685, 0.08673591],
    [ 0.09800301, 0.05302279, 0.09242646],
    [ 0.10194255, 0.05614641, 0.09813162],
    [ 0.10587261, 0.05922941, 0.103854  ],
    [ 0.1097942 , 0.06227277, 0.10959847],
    [ 0.11370826, 0.06527747, 0.11536893],
    [ 0.11761516, 0.06824548, 0.12116393],
    [ 0.12151575, 0.07117741, 0.12698763],
    [ 0.12541095, 0.07407363, 0.1328442 ],
    [ 0.12930083, 0.07693611, 0.13873064],
    [ 0.13317849, 0.07976988, 0.14465095],
    [ 0.13701138, 0.08259683, 0.15060265],
    [ 0.14079223, 0.08542126, 0.15659379],
    [ 0.14452486, 0.08824175, 0.16262484],
    [ 0.14820351, 0.09106304, 0.16869476],
    [ 0.15183185, 0.09388372, 0.17480366],
    [ 0.15540398, 0.09670855, 0.18094993],
    [ 0.15892417, 0.09953561, 0.18713384],
    [ 0.16238588, 0.10236998, 0.19335329],
    [ 0.16579435, 0.10520905, 0.19960847],
    [ 0.16914226, 0.10805832, 0.20589698],
    [ 0.17243586, 0.11091443, 0.21221911],
    [ 0.17566717, 0.11378321, 0.21857219],
    [ 0.17884322, 0.11666074, 0.2249565 ],
    [ 0.18195582, 0.11955283, 0.23136943],
    [ 0.18501213, 0.12245547, 0.23781116],
    [ 0.18800459, 0.12537395, 0.24427914],
    [ 0.19093944, 0.1283047 , 0.25077369],
    [ 0.19381092, 0.13125179, 0.25729255],
    [ 0.19662307, 0.13421303, 0.26383543],
    [ 0.19937337, 0.13719028, 0.27040111],
    [ 0.20206187, 0.14018372, 0.27698891],
    [ 0.20469116, 0.14319196, 0.28359861],
    [ 0.20725547, 0.14621882, 0.29022775],
    [ 0.20976258, 0.14925954, 0.29687795],
    [ 0.21220409, 0.15231929, 0.30354703],
    [ 0.21458611, 0.15539445, 0.31023563],
    [ 0.21690827, 0.15848519, 0.31694355],
    [ 0.21916481, 0.16159489, 0.32366939],
    [ 0.2213631 , 0.16471913, 0.33041431],
    [ 0.22349947, 0.1678599 , 0.33717781],
    [ 0.2255714 , 0.1710185 , 0.34395925],
    [ 0.22758415, 0.17419169, 0.35075983],
    [ 0.22953569, 0.17738041, 0.35757941],
    [ 0.23142077, 0.18058733, 0.3644173 ],
    [ 0.2332454 , 0.18380872, 0.37127514],
    [ 0.2350092 , 0.18704459, 0.3781528 ],
    [ 0.23670785, 0.190297  , 0.38504973],
    [ 0.23834119, 0.19356547, 0.39196711],
    [ 0.23991189, 0.19684817, 0.39890581],
    [ 0.24141903, 0.20014508, 0.4058667 ],
    [ 0.24286214, 0.20345642, 0.4128484 ],
    [ 0.24423453, 0.20678459, 0.41985299],
    [ 0.24554109, 0.21012669, 0.42688124],
    [ 0.2467815 , 0.21348266, 0.43393244],
    [ 0.24795393, 0.21685249, 0.4410088 ],
    [ 0.24905614, 0.22023618, 0.448113  ],
    [ 0.25007383, 0.22365053, 0.45519562],
    [ 0.25098926, 0.22710664, 0.46223892],
    [ 0.25179696, 0.23060342, 0.46925447],
    [ 0.25249346, 0.23414353, 0.47623196],
    [ 0.25307401, 0.23772973, 0.48316271],
    [ 0.25353152, 0.24136961, 0.49001976],
    [ 0.25386167, 0.24506548, 0.49679407],
    [ 0.25406082, 0.2488164 , 0.50348932],
    [ 0.25412435, 0.25262843, 0.51007843],
    [ 0.25404842, 0.25650743, 0.51653282],
    [ 0.25383134, 0.26044852, 0.52286845],
    [ 0.2534705 , 0.26446165, 0.52903422],
    [ 0.25296722, 0.2685428 , 0.53503572],
    [ 0.2523226 , 0.27269346, 0.54085315],
    [ 0.25153974, 0.27691629, 0.54645752],
    [ 0.25062402, 0.28120467, 0.55185939],
    [ 0.24958205, 0.28556371, 0.55701246],
    [ 0.24842386, 0.28998148, 0.56194601],
    [ 0.24715928, 0.29446327, 0.56660884],
    [ 0.24580099, 0.29899398, 0.57104399],
    [ 0.24436202, 0.30357852, 0.57519929],
    [ 0.24285591, 0.30819938, 0.57913247],
    [ 0.24129828, 0.31286235, 0.58278615],
    [ 0.23970131, 0.3175495 , 0.5862272 ],
    [ 0.23807973, 0.32226344, 0.58941872],
    [ 0.23644557, 0.32699241, 0.59240198],
    [ 0.2348113 , 0.33173196, 0.59518282],
    [ 0.23318874, 0.33648036, 0.59775543],
    [ 0.2315855 , 0.34122763, 0.60016456],
    [ 0.23001121, 0.34597357, 0.60240251],
    [ 0.2284748 , 0.35071512, 0.6044784 ],
    [ 0.22698081, 0.35544612, 0.60642528],
    [ 0.22553305, 0.36016515, 0.60825252],
    [ 0.22413977, 0.36487341, 0.60994938],
    [ 0.22280246, 0.36956728, 0.61154118],
    [ 0.22152555, 0.37424409, 0.61304472],
    [ 0.22030752, 0.37890437, 0.61446646],
    [ 0.2191538 , 0.38354668, 0.61581561],
    [ 0.21806257, 0.38817169, 0.61709794],
    [ 0.21703799, 0.39277882, 0.61831922],
    [ 0.21607792, 0.39736958, 0.61948028],
    [ 0.21518463, 0.40194196, 0.62059763],
    [ 0.21435467, 0.40649717, 0.62167507],
    [ 0.21358663, 0.41103579, 0.62271724],
    [ 0.21288172, 0.41555771, 0.62373011],
    [ 0.21223835, 0.42006355, 0.62471794],
    [ 0.21165312, 0.42455441, 0.62568371],
    [ 0.21112526, 0.42903064, 0.6266318 ],
    [ 0.21065161, 0.43349321, 0.62756504],
    [ 0.21023306, 0.43794288, 0.62848279],
    [ 0.20985996, 0.44238227, 0.62938329],
    [ 0.20951045, 0.44680966, 0.63030696],
    [ 0.20916709, 0.45122981, 0.63124483],
    [ 0.20882976, 0.45564335, 0.63219599],
    [ 0.20849798, 0.46005094, 0.63315928],
    [ 0.20817199, 0.46445309, 0.63413391],
    [ 0.20785149, 0.46885041, 0.63511876],
    [ 0.20753716, 0.47324327, 0.63611321],
    [ 0.20722876, 0.47763224, 0.63711608],
    [ 0.20692679, 0.48201774, 0.63812656],
    [ 0.20663156, 0.48640018, 0.63914367],
    [ 0.20634336, 0.49078002, 0.64016638],
    [ 0.20606303, 0.49515755, 0.6411939 ],
    [ 0.20578999, 0.49953341, 0.64222457],
    [ 0.20552612, 0.50390766, 0.64325811],
    [ 0.20527189, 0.50828072, 0.64429331],
    [ 0.20502868, 0.51265277, 0.64532947],
    [ 0.20479718, 0.51702417, 0.64636539],
    [ 0.20457804, 0.52139527, 0.64739979],
    [ 0.20437304, 0.52576622, 0.64843198],
    [ 0.20418396, 0.53013715, 0.64946117],
    [ 0.20401238, 0.53450825, 0.65048638],
    [ 0.20385896, 0.53887991, 0.65150606],
    [ 0.20372653, 0.54325208, 0.65251978],
    [ 0.20361709, 0.5476249 , 0.6535266 ],
    [ 0.20353258, 0.55199854, 0.65452542],
    [ 0.20347472, 0.55637318, 0.655515  ],
    [ 0.20344718, 0.56074869, 0.65649508],
    [ 0.20345161, 0.56512531, 0.65746419],
    [ 0.20349089, 0.56950304, 0.65842151],
    [ 0.20356842, 0.57388184, 0.65936642],
    [ 0.20368663, 0.57826181, 0.66029768],
    [ 0.20384884, 0.58264293, 0.6612145 ],
    [ 0.20405904, 0.58702506, 0.66211645],
    [ 0.20431921, 0.59140842, 0.66300179],
    [ 0.20463464, 0.59579264, 0.66387079],
    [ 0.20500731, 0.60017798, 0.66472159],
    [ 0.20544449, 0.60456387, 0.66555409],
    [ 0.20596097, 0.60894927, 0.66636568],
    [ 0.20654832, 0.61333521, 0.66715744],
    [ 0.20721003, 0.61772167, 0.66792838],
    [ 0.20795035, 0.62210845, 0.66867802],
    [ 0.20877302, 0.62649546, 0.66940555],
    [ 0.20968223, 0.63088252, 0.6701105 ],
    [ 0.21068163, 0.63526951, 0.67079211],
    [ 0.21177544, 0.63965621, 0.67145005],
    [ 0.21298582, 0.64404072, 0.67208182],
    [ 0.21430361, 0.64842404, 0.67268861],
    [ 0.21572716, 0.65280655, 0.67326978],
    [ 0.21726052, 0.65718791, 0.6738255 ],
    [ 0.21890636, 0.66156803, 0.67435491],
    [ 0.220668  , 0.66594665, 0.67485792],
    [ 0.22255447, 0.67032297, 0.67533374],
    [ 0.22458372, 0.67469531, 0.67578061],
    [ 0.22673713, 0.67906542, 0.67620044],
    [ 0.22901625, 0.6834332 , 0.67659251],
    [ 0.23142316, 0.68779836, 0.67695703],
    [ 0.23395924, 0.69216072, 0.67729378],
    [ 0.23663857, 0.69651881, 0.67760151],
    [ 0.23946645, 0.70087194, 0.67788018],
    [ 0.24242624, 0.70522162, 0.67813088],
    [ 0.24549008, 0.70957083, 0.67835215],
    [ 0.24863372, 0.71392166, 0.67854868],
    [ 0.25187832, 0.71827158, 0.67872193],
    [ 0.25524083, 0.72261873, 0.67887024],
    [ 0.25870947, 0.72696469, 0.67898912],
    [ 0.26229238, 0.73130855, 0.67907645],
    [ 0.26604085, 0.73564353, 0.67914062],
    [ 0.26993099, 0.73997282, 0.67917264],
    [ 0.27397488, 0.74429484, 0.67917096],
    [ 0.27822463, 0.74860229, 0.67914468],
    [ 0.28264201, 0.75290034, 0.67907959],
    [ 0.2873016 , 0.75717817, 0.67899164],
    [ 0.29215894, 0.76144162, 0.67886578],
    [ 0.29729823, 0.76567816, 0.67871894],
    [ 0.30268199, 0.76989232, 0.67853896],
    [ 0.30835665, 0.77407636, 0.67833512],
    [ 0.31435139, 0.77822478, 0.67811118],
    [ 0.3206671 , 0.78233575, 0.67786729],
    [ 0.32733158, 0.78640315, 0.67761027],
    [ 0.33437168, 0.79042043, 0.67734882],
    [ 0.34182112, 0.79437948, 0.67709394],
    [ 0.34968889, 0.79827511, 0.67685638],
    [ 0.35799244, 0.80210037, 0.67664969],
    [ 0.36675371, 0.80584651, 0.67649539],
    [ 0.3759816 , 0.80950627, 0.67641393],
    [ 0.38566792, 0.81307432, 0.67642947],
    [ 0.39579804, 0.81654592, 0.67656899],
    [ 0.40634556, 0.81991799, 0.67686215],
    [ 0.41730243, 0.82318339, 0.67735255],
    [ 0.4285828 , 0.82635051, 0.6780564 ],
    [ 0.44012728, 0.82942353, 0.67900049],
    [ 0.45189421, 0.83240398, 0.68021733],
    [ 0.46378379, 0.83530763, 0.6817062 ],
    [ 0.47573199, 0.83814472, 0.68347352],
    [ 0.48769865, 0.84092197, 0.68552698],
    [ 0.49962354, 0.84365379, 0.68783929],
    [ 0.5114027 , 0.8463718 , 0.69029789],
    [ 0.52301693, 0.84908401, 0.69288545],
    [ 0.53447549, 0.85179048, 0.69561066],
    [ 0.54578602, 0.8544913 , 0.69848331],
    [ 0.55695565, 0.85718723, 0.70150427],
    [ 0.56798832, 0.85987893, 0.70468261],
    [ 0.57888639, 0.86256715, 0.70802931],
    [ 0.5896541 , 0.8652532 , 0.71154204],
    [ 0.60028928, 0.86793835, 0.71523675],
    [ 0.61079441, 0.87062438, 0.71910895],
    [ 0.62116633, 0.87331311, 0.72317003],
    [ 0.63140509, 0.87600675, 0.72741689],
    [ 0.64150735, 0.87870746, 0.73185717],
    [ 0.65147219, 0.8814179 , 0.73648495],
    [ 0.66129632, 0.8841403 , 0.74130658],
    [ 0.67097934, 0.88687758, 0.74631123],
    [ 0.68051833, 0.88963189, 0.75150483],
    [ 0.68991419, 0.89240612, 0.75687187],
    [ 0.69916533, 0.89520211, 0.76241714],
    [ 0.70827373, 0.89802257, 0.76812286],
    [ 0.71723995, 0.90086891, 0.77399039],
    [ 0.72606665, 0.90374337, 0.7800041 ],
    [ 0.73475675, 0.90664718, 0.78615802],
    [ 0.74331358, 0.90958151, 0.79244474],
    [ 0.75174143, 0.91254787, 0.79884925],
    [ 0.76004473, 0.91554656, 0.80536823],
    [ 0.76827704, 0.91856549, 0.81196513],
    [ 0.77647029, 0.921603  , 0.81855729],
    [ 0.78462009, 0.92466151, 0.82514119],
    [ 0.79273542, 0.92773848, 0.83172131],
    [ 0.8008109 , 0.93083672, 0.83829355],
    [ 0.80885107, 0.93395528, 0.84485982],
    [ 0.81685878, 0.9370938 , 0.85142101],
    [ 0.82483206, 0.94025378, 0.8579751 ],
    [ 0.83277661, 0.94343371, 0.86452477],
    [ 0.84069127, 0.94663473, 0.87106853],
    [ 0.84857662, 0.9498573 , 0.8776059 ],
    [ 0.8564431 , 0.95309792, 0.88414253],
    [ 0.86429066, 0.95635719, 0.89067759],
    [ 0.87218969, 0.95960708, 0.89725384]
]
mako = ListedColormap(mako, name='mako')
mako_r = mako.reversed()
mpl.colormaps.register(cmap=mako)
mpl.colormaps.register(cmap=mako_r)