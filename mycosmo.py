import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy import constants
from astropy import constants as const
from astropy import units as u
import pycamb


class lcdm():
	""" class to encapsulate a flat, lambda-cold-dark-matter (lcdm) cosmology. """
	def __init__(self, params): #omr=0.0, omb=0.05, omc=0.25, oml=0.7, H0=70. ):
        
		assert (1 - np.abs(params['omegab'] + params['omegac'] + params['omegav']) <= 0.05)


		self.omegar = 0. # TODO: add radiation density
		self.omegab = params['omegab']
		self.omegac = params['omegac']
		self.omegav = params['omegav']
		self.H0     = params['H0']
		self.h      = self.H0 / 100.

		self.omegam = params['omegab']+params['omegac']

		self.zvec = np.concatenate( [np.linspace(0., 20., 500., endpoint=False),
		                     np.linspace( 20., 200., 200., endpoint=False),
		                     np.linspace(200., 1500., 100)] )
		# self.xvec = np.array( [ integrate.quad( lambda z : (const.c.value * 1.e-3) / self.H_z(z), 0., zmax )[0] for zmax in self.zvec ] )
		self.xvec = np.array( [ integrate.quad( lambda z : (const.c.to('km/s').value) / self.H_z(z), 0., zmax )[0] for zmax in self.zvec ] )

		self.zmin = np.min(self.zvec)
		self.zmax = np.max(self.zvec)
		self.xmin = np.min(self.xvec)
		self.xmax = np.max(self.xvec)

		self.spl_x_z = interpolate.UnivariateSpline( self.zvec, self.xvec, k=1, s=0 )
		self.spl_z_x = interpolate.UnivariateSpline( self.xvec, self.zvec, k=1, s=0 )

		self.chi_rec = self.spl_x_z(1090.)

		# Evaluating Matter PS
		max_kappa = 80.
		kappa = np.logspace(-3,np.log10(max_kappa),50)
		z = np.logspace(-3,1,num=50)
		z = np.insert(z,0,0.)
		redshifts = z[::-1]

		mps = pycamb.matter_power(redshifts, k=kappa, **params)
		mps = np.asarray(mps)

		@np.vectorize
		def pk_hi(k, z):
			return mps[1,-1,z]*(k/mps[0,-1,z])**-3

		k_hi = np.linspace(mps[0,-1,0],2000,1000)[1:]
		mps_hi = np.zeros((2, k_hi.size, redshifts.size))

		for i in xrange(0, redshifts.size):
			mps_hi[1,:,i] = pk_hi(k_hi,i)
			mps_hi[0,:,i] = k_hi

		mps_ext_hi = np.hstack((mps, mps_hi))
		self.pkz   = interpolate.RectBivariateSpline(mps_ext_hi[0,:,0], z, mps_ext_hi[1,:,::-1], kx=3, ky=2, s=0)    # (Mpc/h)^3

	
	def t_z(self, z):
		""" returns the age of the Universe (in Gyr) at redshift z. """

		# da/dt / a = H_a
		# da / H_a / a = dt
		# /int_{a=0}^{a(z)} da / H_a / a = t
		# H0 = km/s/Mpc * 1Mpc/1e6pc * 1e3m/km * 3.08e16pc / m
		# 1Mpc = 3.25e6 ly
		return integrate.quad( lambda a : 1. / (self.H_a(a) / 3.08e19) / a / (365*24.*60.*60.), 1.e-10, 1./(1+z) )[0]/1.e9

	def x_z(self, z):
		""" returns the conformal distance (in Mpc) to redshift z. """
		assert( np.all( z >= self.zmin ) )
		assert( np.all( z <= self.zmax ) )
		return self.spl_x_z(z)

	def z_x(self, x):
		""" returns the redshift z at conformal distance x (in Mpc)."""
		assert( np.all( x >= self.xmin ) )
		assert( np.all( x <= self.xmax ) )
		return self.spl_z_x(x)

	def H_a(self, a):
		""" returns the hubble factor at scale factor a=1/(1+z). """
		return self.H0 * np.sqrt(self.omegav + self.omegam * a**(-3) + self.omegar * a**(-4))

	def H_z(self, z):
		""" returns the hubble factor at redshift z. """
		return self.H0 * (self.omegam * (1.+z)**3. + (1.- self.omegam))**0.5

		#return self.H_a( 1./(1.+z) )

	def H_x(self, x):
		""" returns the hubble factor at conformal distance x (in Mpc). """
		return self.H_z( self.z_x(x) )

	def G_z(self, z):
		""" returns the growth factor at redshift z (Eq. 7.77 of Dodelson). """
		if np.isscalar(z) or (np.size(z) == 1):
			return 2.5 * self.omm * self.H_a(1./(1.+z)) / self.H0 * integrate.quad( lambda a : ( self.H0 / (a * self.H_a(a)) )**3, 0, 1./(1.+z) )[0] 
		else:
			return [ self.G_z(tz) for tz in z ]

	def G_x(self, x):
		""" returns the growth factor at conformal distance x (in Mpc) (Eq. 7.77 of Dodelson). """
		return self.G_z( self.z_x(x) )

	def Dv_mz(self, z):
		""" returns the virial overdensity w.r.t. the mean matter density redshift z. based on
		       * Bryan & Norman (1998) ApJ, 495, 80.
		       * Hu & Kravtsov (2002) astro-ph/0203169 Eq. C6. """
		den = self.oml + self.omm * (1.0+z)**3 + self.omr * (1.0+z)**4
		omm = self.omm * (1.0+z)**3 / den

		omr = self.omr * (1.0+z)**4 / den
		assert(omr < 1.e-2) # sanity check that omr is negligible at this redshift.

		return (18.*np.pi**2 + 82.*(omm - 1.) - 39*(omm - 1.)**2) / omm

	def aeq_lm(self):
		""" returns the scale factor at lambda - matter equality. """
		return 1. / ( self.oml / self.omm )**(1./3.)

	def magnification_integral(self, z, dndz): # !!! WITHOUT THE (\alpha - 1) FACTOR !!!
		if np.isscalar(z) or (np.size(z) == 1):
			def integrand(zprime, z, dndz):
			    return  (1.-self.x_z(z)/self.x_z(zprime)) * dndz(zprime)
			z_ = np.linspace(z, 10, 1000)
			return (3.*self.omegam*self.H0**2. * (u.km).to(u.Mpc)) /(2*const.c.value * self.H_z(z)) * (1.+z) * self.x_z(z)*(u.Mpc).to(u.m) * integrate.simps(integrand(z_, z, dndz), x=z_)
		else:
			return [ self.magnification_integral(tz, dndz) for tz in z ]

	def w_kappa_cmb(self, z):
		return (3.*self.omegam*self.H0**2 * (u.km).to(u.Mpc))/(2.*const.c.to('Mpc/s').value * self.H_z(z)) * self.x_z(z) * (1.+z) * ((self.chi_rec-self.x_z(z))/self.chi_rec)

	def w_cl(self, z, dndz, b=1.):
		return b * dndz(z)

	def w_mu(self, z, dndz, alpha=1.): # !!! WITHOUT THE (\alpha - 1) FACTOR !!!
		return self.magnification_integral(z, dndz)

	# Power Spectra
	def power_spectra_clustering_at_z_and_l(self, z, ell, dndz1, dndz2=None, b1=1, b2=1):
		if np.isscalar(z) or (np.size(z) == 1):
			if dndz2 is None:
			    return (self.H_z(z) * (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * self.w_cl(z, dndz1, b=b1)**2 * self.pkz(ell/(self.x_z(z)*self.h), z)
			else:
			    return (self.H_z(z) * (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * self.w_cl(z, dndz1, b=b1) * self.w_cl(z, dndz2, b=b2) * self.pkz(ell/(self.x_z(z)*self.h), z)
		else:
			return [ self.power_spectra_clustering_at_z_and_l(tz, ell, dndz1, dndz2=dndz2, b1=b1, b2=b2) for tz in z ]

	def power_spectra_clustering(self, dndz1, dndz2=None, b1=1, b2=1, lmax=2500):
	    cl_spline   = []

	    l_log = np.logspace(np.log10(2),np.log10(lmax),20)
	    l_log = np.unique(l_log.astype(int))
	    good  = np.where(l_log<3)
	    l_log = l_log[np.where(l_log>20)]
	    l_log = np.insert(l_log,0,2)

	    zmin = 0.01
	    zmax = 7

	    delta_z = 0.05
	    N_step = (zmax-zmin)/delta_z
	    zeta = np.linspace(zmin,zmax,N_step)

	    # Logspace in ell and linspace in z    
	    for ell in l_log:
	        
	        cl_integrand_spline_points = self.power_spectra_clustering_at_z_and_l(zeta, ell, dndz1=dndz1, dndz2=dndz2, b1=b1, b2=b2)
	        cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

	        zeta_new = np.insert(zeta,0,0.)

	        cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

	    cl_spline = np.asarray(cl_spline)/self.h**3

	    cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

	    return cl(np.arange(lmax+1))

	def power_spectra_crossterm_at_z_and_l(self, z, ell, dndz1, dndz2, b1=1, alpha2=1):
		# dndz1 is clustering
		# dndz2 is magnification
		if np.isscalar(z) or (np.size(z) == 1):
			return (self.H_z(z)* (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * self.w_cl(z, dndz1, b=b1)*self.w_mu(z, dndz2, alpha=alpha2) * self.pkz(ell/(self.x_z(z)*self.h),z)
		else:
			return [ self.power_spectra_crossterm_at_z_and_l(tz, ell, dndz1, dndz2=dndz2, b1=b1, alpha2=alpha2) for tz in z ]

	def power_spectra_crossterm(self, dndz1, dndz2, b1=1, alpha2=1, lmax=2500):
	    cl_spline   = []

	    l_log = np.logspace(np.log10(2),np.log10(lmax),20)
	    l_log = np.unique(l_log.astype(int))
	    good  = np.where(l_log<3)
	    l_log = l_log[np.where(l_log>20)]
	    l_log = np.insert(l_log,0,2)

	    zmin = 0.01
	    zmax = 10

	    delta_z = 0.05
	    N_step = (zmax-zmin)/delta_z
	    zeta = np.linspace(zmin,zmax,N_step)

	    # Logspace in ell and linspace in z    
	    for ell in l_log:
	        
	        cl_integrand_spline_points = self.power_spectra_crossterm_at_z_and_l(zeta, ell, dndz1=dndz1, dndz2=dndz2, b1=b1, alpha2=alpha2)
	        cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

	        zeta_new = np.insert(zeta,0,0.)

	        cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

	    cl_spline = np.asarray(cl_spline)/(self.h)**3

	    cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

	    return cl(np.arange(lmax+1))

	def power_spectra_magnification_at_z_and_l(self, z, ell, dndz1, dndz2=None, alpha1=1, alpha2=1):
		if np.isscalar(z) or (np.size(z) == 1):
		    if dndz2 is None:
		        return (self.H_z(z)* (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * self.w_mu(z, dndz1, alpha=alpha1)**2 * self.pkz(ell/(self.x_z(z)*self.h),z)
		    else:
		        return (self.H_z(z)* (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * self.w_mu(z, dndz1, alpha=alpha1)*self.w_mu(z, dndz2, alpha=alpha2) * self.pkz(ell/(self.x_z(z)*self.h),z)
		else:
			return [ self.power_spectra_magnification_at_z_and_l(tz, ell, dndz1, dndz2=dndz2, alpha1=alpha1, alpha2=alpha2) for tz in z ]

	def power_spectra_magnification(self, dndz1, dndz2, alpha1=1, alpha2=1, lmax=2500):
	    cl_spline   = []

	    l_log = np.logspace(np.log10(2),np.log10(lmax),20)
	    l_log = np.unique(l_log.astype(int))
	    good  = np.where(l_log<3)
	    l_log = l_log[np.where(l_log>20)]
	    l_log = np.insert(l_log,0,2)

	    zmin = 0.01
	    zmax = 7

	    delta_z = 0.05
	    N_step = (zmax-zmin)/delta_z
	    zeta = np.linspace(zmin,zmax,N_step)

	    # Logspace in ell and linspace in z    
	    for ell in l_log:
	        
	        cl_integrand_spline_points = self.power_spectra_magnification_at_z_and_l(zeta, ell, dndz1=dndz1, dndz2=dndz2, alpha1=alpha1, alpha2=alpha2)
	        cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

	        zeta_new = np.insert(zeta,0,0.)

	        cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

	    cl_spline = np.asarray(cl_spline)/(self.h)**3

	    cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

	    return cl(np.arange(lmax+1))

	def power_spectra_kg_at_z_and_l(self, z, ell, dndz, b=1):
		if np.isscalar(z) or (np.size(z) == 1):
			return (self.H_z(z) * (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * self.w_cl(z, dndz, b=b) * self.w_kappa_cmb(z) * self.pkz(ell/(self.x_z(z)*self.h), z)
		else:
			return [ self.power_spectra_kg_at_z_and_l(tz, ell, dndz, b=b) for tz in z ]

	def power_spectra_kg(self, dndz, b=1, lmax=2500):
	    cl_spline   = []

	    l_log = np.logspace(np.log10(2),np.log10(lmax),20)
	    l_log = np.unique(l_log.astype(int))
	    good  = np.where(l_log<3)
	    l_log = l_log[np.where(l_log>20)]
	    l_log = np.insert(l_log,0,2)

	    zmin = 0.01
	    zmax = 7

	    delta_z = 0.05
	    N_step = (zmax-zmin)/delta_z
	    zeta = np.linspace(zmin,zmax,N_step)

	    # Logspace in ell and linspace in z    
	    for ell in l_log:
	        
	        cl_integrand_spline_points = self.power_spectra_kg_at_z_and_l(zeta, ell, dndz=dndz, b=b)
	        cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

	        zeta_new = np.insert(zeta,0,0.)

	        cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

	    cl_spline = np.asarray(cl_spline)/(self.h)**3

	    cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

	    return cl(np.arange(lmax+1))

	def power_spectra_kmu_at_z_and_l(self, z, ell, dndz): # !!! WITHOUT THE (\alpha - 1) FACTOR !!!
		if np.isscalar(z) or (np.size(z) == 1):
			return (self.H_z(z) * (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * self.w_kappa_cmb(z) * self.w_mu(z, dndz) * self.pkz(ell/(self.x_z(z)*self.h), z)
		else:
			return [ self.power_spectra_kmu_at_z_and_l(tz, ell, dndz) for tz in z ]

	def power_spectra_kmu(self, dndz, lmax=2500): # !!! WITHOUT THE (\alpha - 1) FACTOR !!!
	    cl_spline   = []

	    l_log = np.logspace(np.log10(2),np.log10(lmax),20)
	    l_log = np.unique(l_log.astype(int))
	    good  = np.where(l_log<3)
	    l_log = l_log[np.where(l_log>20)]
	    l_log = np.insert(l_log,0,2)

	    zmin = 0.01
	    zmax = 7

	    delta_z = 0.05
	    N_step = (zmax-zmin)/delta_z
	    zeta = np.linspace(zmin,zmax,N_step)

	    # Logspace in ell and linspace in z    
	    for ell in l_log:
	        
	        cl_integrand_spline_points = self.power_spectra_kmu_at_z_and_l(zeta, ell, dndz)
	        cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

	        zeta_new = np.insert(zeta,0,0.)

	        cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

	    cl_spline = np.asarray(cl_spline)/(self.h)**3

	    cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

	    return cl(np.arange(lmax+1))

	def power_spectra_gg(self, dndz, lmax=2500):
		cl_spline   = []

		l_log = np.logspace(np.log10(2),np.log10(lmax),20)
		l_log = np.unique(l_log.astype(int))
		good  = np.where(l_log<3)
		l_log = l_log[np.where(l_log>20)]
		l_log = np.insert(l_log,0,2)

		zmin = 0.01
		zmax = 7

		delta_z = 0.05
		N_step = (zmax-zmin)/delta_z
		zeta = np.linspace(zmin,zmax,N_step)

		# Logspace in ell and linspace in z    
		for ell in l_log:
		    
		    cl_integrand_spline_points = self.power_spectra_clustering_at_z_and_l(zeta, ell, dndz)
		    cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

		    zeta_new = np.insert(zeta,0,0.)

		    cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

		cl_spline = np.asarray(cl_spline)/(self.h)**3

		cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

		return cl(np.arange(lmax+1))

	def power_spectra_gmu(self, dndz, lmax=2500): # CROSS TERM WITHOUT A FACTOR 2 (and (\alpha-1) and b)
		cl_spline   = []

		l_log = np.logspace(np.log10(2),np.log10(lmax),20)
		l_log = np.unique(l_log.astype(int))
		good  = np.where(l_log<3)
		l_log = l_log[np.where(l_log>20)]
		l_log = np.insert(l_log,0,2)

		zmin = 0.01
		zmax = 7

		delta_z = 0.05
		N_step = (zmax-zmin)/delta_z
		zeta = np.linspace(zmin,zmax,N_step)

		# Logspace in ell and linspace in z    
		for ell in l_log:
		    
		    cl_integrand_spline_points = self.power_spectra_crossterm_at_z_and_l(zeta, ell, dndz, dndz, alpha2=2)
		    cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

		    zeta_new = np.insert(zeta,0,0.)

		    cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

		cl_spline = np.asarray(cl_spline)/(self.h)**3

		cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

		return cl(np.arange(lmax+1))

	def power_spectra_mumu(self, dndz, lmax=2500):
		cl_spline   = []

		l_log = np.logspace(np.log10(2),np.log10(lmax),20)
		l_log = np.unique(l_log.astype(int))
		good  = np.where(l_log<3)
		l_log = l_log[np.where(l_log>20)]
		l_log = np.insert(l_log,0,2)

		zmin = 0.01
		zmax = 7

		delta_z = 0.05
		N_step = (zmax-zmin)/delta_z
		zeta = np.linspace(zmin,zmax,N_step)

		# Logspace in ell and linspace in z    
		for ell in l_log:
		    
		    cl_integrand_spline_points = self.power_spectra_magnification_at_z_and_l(zeta, ell, dndz, alpha1=2)
		    cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

		    zeta_new = np.insert(zeta,0,0.)

		    cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

		cl_spline = np.asarray(cl_spline)/(self.h)**3

		cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

		return cl(np.arange(lmax+1))

	def power_spectra_gg_tot_at_z_and_l(self, z, ell, dndz, b=1, alpha=1):
		if np.isscalar(z) or (np.size(z) == 1):
		    return (self.H_z(z) * (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * (self.w_cl(z, dndz, b=b) + (alpha-1)*self.w_mu(z, dndz, alpha=alpha))**2 * self.pkz(ell/(self.x_z(z)*self.h), z)
		else:
			return [ self.power_spectra_gg_tot_at_z_and_l(tz, ell, dndz, b=b, alpha=alpha) for tz in z ]

	def power_spectra_gg_tot(self, dndz, b=1, alpha=1, lmax=2500):
		cl_spline   = []

		l_log = np.logspace(np.log10(2),np.log10(lmax),20)
		l_log = np.unique(l_log.astype(int))
		good  = np.where(l_log<3)
		l_log = l_log[np.where(l_log>20)]
		l_log = np.insert(l_log,0,2)

		zmin = 0.01
		zmax = 7

		delta_z = 0.05
		N_step = (zmax-zmin)/delta_z
		zeta = np.linspace(zmin,zmax,N_step)

		# Logspace in ell and linspace in z    
		for ell in l_log:
		    
		    cl_integrand_spline_points = self.power_spectra_gg_tot_at_z_and_l(zeta, ell, dndz, b=b, alpha=alpha)
		    cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

		    zeta_new = np.insert(zeta,0,0.)

		    cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

		cl_spline = np.asarray(cl_spline)/(self.h)**3

		cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

		return cl(np.arange(lmax+1))

	def power_spectra_kg_tot_at_z_and_l(self, z, ell, dndz, b=1, alpha=1):
		if np.isscalar(z) or (np.size(z) == 1):
		    return (self.H_z(z) * (u.km).to(u.Mpc))/(self.x_z(z)**2 * const.c.to('Mpc/s').value) * self.w_kappa_cmb(z) * (self.w_cl(z, dndz, b=b) + (alpha-1)*self.w_mu(z, dndz, alpha=alpha)) * self.pkz(ell/(self.x_z(z)*self.h), z)
		else:
			return [ self.power_spectra_kg_tot_at_z_and_l(tz, ell, dndz, b=b, alpha=alpha) for tz in z ]

	def power_spectra_kg_tot(self, dndz, b=1, alpha=1, lmax=2500):
		cl_spline   = []

		l_log = np.logspace(np.log10(2),np.log10(lmax),20)
		l_log = np.unique(l_log.astype(int))
		good  = np.where(l_log<3)
		l_log = l_log[np.where(l_log>20)]
		l_log = np.insert(l_log,0,2)

		zmin = 0.01
		zmax = 7

		delta_z = 0.05
		N_step = (zmax-zmin)/delta_z
		zeta = np.linspace(zmin,zmax,N_step)

		# Logspace in ell and linspace in z    
		for ell in l_log:
		    
		    cl_integrand_spline_points = self.power_spectra_kg_tot_at_z_and_l(zeta, ell, dndz, b=b, alpha=alpha)
		    cl_integrand_spline_points = np.insert(cl_integrand_spline_points,0,0.)

		    zeta_new = np.insert(zeta,0,0.)

		    cl_spline.append(integrate.simps(cl_integrand_spline_points, x=zeta_new))

		cl_spline = np.asarray(cl_spline)/(self.h)**3

		cl = interpolate.InterpolatedUnivariateSpline(l_log, cl_spline, k=2)

		return cl(np.arange(lmax+1))

# Redshift distributions
def cond_prob(zp, z, sigma=0.26, bias=0.):
	return np.exp(-((zp-z-bias)**2./(2.*(sigma*(1+zp))**2)))/(2.*np.pi*(sigma*(1+zp))**2)**0.5

def get_dNdz_spline(dNdz, z):
	return interpolate.interp1d(z, dNdz, bounds_error=False, fill_value=0.)

def convolve_window(z, sigma=0.26, bias=0., z_min=0., z_max=10.):
	if np.isscalar(z) or (np.size(z) == 1):
		zp = np.linspace(z_min, z_max, 1000)
		return integrate.simps(cond_prob(zp, z, sigma=sigma, bias=bias), x=zp)
	else:
		return np.asarray([ convolve_window(tz, sigma=sigma, z_min=z_min, z_max=z_max) for tz in z ])

def catalog2dndz(z_cat, zbins=(0., 10.), sigma=0.26, bias=0, nbins=30):
	'Returns the interpolation object of a dN/dz given the array with the redshifts'
	dNdz, edges  = np.histogram(z_cat, nbins, density=True)
	bins_ 		 = (edges[:-1] + edges[1:])/2. 
	dNdz_cat     = get_dNdz_spline(dNdz, bins_)

	z        = np.linspace(0., 10, 1000)
	phi      = dNdz_cat(z) * convolve_window(z, sigma=sigma, bias=bias, z_min=zbins[0], z_max=zbins[1])
	phi_norm = integrate.simps(phi, x=z)
	phi     /= phi_norm

	return interpolate.interp1d(z, phi, bounds_error=False, fill_value=0.)

def dndz2phi(dNdz, z, z_bins=(0., 10.), sigma=0.26, bias=0, nbins=30):
	dNdz_cat = get_dNdz_spline(dNdz, z)

	z_       = np.linspace(0., 10, 1000)
	phi      = dNdz_cat(z) * convolve_window(z_, sigma=sigma, bias=bias, z_min=zbins[0], z_max=zbins[1])
	phi_norm = integrate.simps(phi, x=z_)
	phi     /= phi_norm

	return interpolate.interp1d(z_, phi, bounds_error=False, fill_value=0.)

