import numpy as np
import sys, os
import emcee
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy import interpolate, optimize
from getdist import plots, MCSamples
from multiprocessing import Pool
from IPython import embed
from astropy.table import Table, Column
from astropy.io import fits

rc('text',usetex=True)
rc('font',**{'family':'serif','serif':['Computer Modern']})
plt.rcParams['axes.linewidth']  = 2.
plt.rcParams['axes.labelsize']  = 20
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['legend.fontsize']  = 12
plt.rcParams['legend.frameon']  = False

plt.rcParams['xtick.major.width'] = 1
plt.rcParams['ytick.major.width'] = 1
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.minor.width'] = 1

@np.vectorize
def flux_at_z_and_lambda(z, lam_um, SED):
	'''
	Gives the *un-normalized* flux at a given observed wavelength lambda and
	redshift z (and SED). 
	! lambda in [micron] !
	! SED is spline object in [erg/s/Hz]
	'''
	return SED(lam_um/(1+z)) #/ ((1+z)*4*np.pi*d_l(z)**2) /10**3 #*cosmo.luminosity_distance(z).to("cm").value**2) / 10**3 # -> mJy

def get_SED_spline(file_SED, norm=1e14, lam_um_min=8., lam_um_max=1000, fill_value=0.):
	'''
	Read SED from file and returns spline object.
	It assumes first column is wavelength in micron, and second one is SED in erg/s/Hz
	'''
	lam_, sed_ = np.loadtxt(file_SED, unpack=True)
	# lam,  sed  = lam_[(lam_ >= lam_um_min) & (lam_ <= lam_um_max)], sed_[(lam_ >= lam_um_min) & (lam_ <= lam_um_max)]
	# plt.loglog(lam_, sed_, 'k', label='SMM-J2135')
	# plt.xlabel(r'$\lambda \, [\mu$m]')
	# plt.axvline(250, color='r', label=r'$250 \mu$m')
	# plt.axvline(350, color='g', label=r'$350 \mu$m')
	# plt.axvline(500, color='b', label=r'$500 \mu$m')
	# plt.ylabel('SED [erg/s/Hz]')
	# plt.legend()
	# plt.show()
	return interpolate.interp1d(lam_, sed_*norm, kind='slinear', fill_value=fill_value, bounds_error=False)
	# return interpolate.UnivariateSpline(lam_, sed_*norm)

def lnLike(theta, flux_data, err_flux_data, SED):
	z, A = theta
	flux_model = SED([250.,350.,500.]/(1+z))
	# print flux_data
	# print A * flux_model
	# print err_flux_data
	# print ''
	return -np.sum(((flux_data - A * flux_model)/err_flux_data)**2)

def lnPr(theta, prior='jeffreys'): 
	z, A = theta
	if z < 0. or z > 10. or A < 0.:
		return -np.inf
	else:
		if prior == 'jeffreys':
			return -np.log(A)
		elif prior == 'flat':
			return 0.

def lnPost(theta, flux_data, err_flux_data, SED, prior='jeffreys'):
	lp = lnPr(theta, prior=prior)
	if not np.isfinite(lp):
	    return -np.inf
	return lp + lnLike(theta, flux_data, err_flux_data, SED)

def go_mcmc(flux_data, err_flux_data, SED, prior='jeffreys', p0=None, nwalkers=150, nsteps=300, nburnin=50):
	'''
	Runs MCMC analysis.
	'''
	ndim = 2

	if p0 is None:
		p0 = [1. for i in xrange(ndim)]
	
	nll = lambda *args: -lnLike(*args)

	# # Guessing starting point in parameters space
	try:
		result = optimize.minimize(nll, p0, args=(flux_data, err_flux_data, SED), tol=1e-1, options={'disp': False})
		if any(t > 0. for t in result["x"]):
			p0 = np.abs([result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)])
			# print result["x"]
		else:
			fad
	except:
		# p0 = [[p0 + 1e-4*np.random.randn(ndim)] for i in range(nwalkers)]
		p0 = [[np.random.rand(), 3.*np.random.rand()] for i in range(nwalkers)]

	# Initialize the sampler with the chosen specs
	_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnPost, args=(flux_data, err_flux_data, SED, prior))#, pool=pool)

	# Run nburnin steps as a burn-in.
	pos, prob, state = _sampler.run_mcmc(p0, nburnin)

	# Reset the chain to remove the burn-in samples
	_sampler.reset()

	# Starting from the final position in the burn-in chain, sample for nsteps steps.
	_sampler.run_mcmc(pos, nsteps, rstate0=state)


	return _sampler

def like_contours_2d(flux_data, err_flux_data, SED, nstep=100):
    zgrid = np.linspace(0,5,nstep)
    Agrid = np.linspace(0,5,nstep)
    chi2grid = np.empty((zgrid.size,Agrid.size))
    for iA, A in enumerate(Agrid):
        for (iz, z) in enumerate(zgrid):
            chi2grid[iz, iA] = -lnLike((z,A), flux_data, err_flux_data, SED)
    chi2grid -= chi2grid.min()
    plt.contour(Agrid, zgrid, chi2grid, levels=(1,4,9))
    plt.ylabel("$z_{ph}$ ")
    plt.xlabel("$A$ ")

def plot_chains(chains, labels, filename=None):
	'''
	Function that plots walker position in parameter space vs. step number.
	It may be helpful for checking chain convergence.
	'''
	import matplotlib.pyplot as plt

	nparams = len(labels)

	if chains.ndim == 2: # Only 1 parameter
		assert (nparams == 1)
	else:
		assert (nparams == chains.shape[2])

	fig = plt.figure()
	for i in xrange(nparams):
		ax = fig.add_subplot(nparams, 1, i+1) 
		if nparams == 1:
			for walker in xrange(chains.shape[0]):
				ax.plot(chains[walker, :], color='grey')
		else:
			for walker in xrange(chains.shape[0]):
				ax.plot(chains[walker, :, i],color='grey')
		ax.set_ylabel(labels[i])
	ax.set_xlabel('Step Number')

def plot_bestfit(z_bf, A_bf, flux_data, err_flux_data, SED):
	lambda_ = np.linspace(1e2,1e3,100)
	# plt.loglog(lambda_, SED(lambda_), 'k', label='SMM-J2135')			
	plt.loglog(lambda_, A_bf*flux_at_z_and_lambda(z_bf, lambda_, SED), 'grey', label='z={0:.2f}, A={1:.3f}'.format(z_ph, A_bf))			
	plt.errorbar(250, flux_data[0], yerr=err_flux_data[0], fmt='o', label='$250\mu$m')
	plt.errorbar(350, flux_data[1], yerr=err_flux_data[1], fmt='o', label='$350\mu$m')
	plt.errorbar(500, flux_data[2], yerr=err_flux_data[2], fmt='o', label='$500\mu$m')
	plt.ylabel(r'SED [erg/s/Hz] $(\times 10^{16})$')
	plt.xlabel(r'$\lambda [\mu$m]')
	plt.legend(loc='best')
	plt.xlim([1e2,6e2])
	plt.ylim([1e1,1e2])
	# plt.show()

def do_analysis(ids):
	# print ids
	flux_data     = np.asarray([data[data['ID']==ids]['s250'][0], data[data['ID']==ids]['s350'][0], data[data['ID']==ids]['s500'][0]])
	err_flux_data = np.asarray([data[data['ID']==ids]['err_s250'][0], data[data['ID']==ids]['err_s350'][0], data[data['ID']==ids]['err_s500'][0]])
	# print err_flux_data

	# Goin' MCMC
	sampler  = go_mcmc(flux_data, err_flux_data, SED, prior=prior, p0=p0, nwalkers=nwalkers, nsteps=nsteps, nburnin=nburnin)
	samples  = sampler.flatchain
	samps_MC = MCSamples(samples=samples, names=par_names, labels=par_label)

	# Mean chains acceptance fraction
	f = np.mean(sampler.acceptance_fraction)

	# Get p(z_ph) 1D densities
	p_z = samps_MC.get1DDensity('z_ph').P
	_z_ = samps_MC.get1DDensity('z_ph').x

	# Peak of p(z) posterior
	z_ph_max = _z_[np.argmax(p_z)]

	# Draw a z_ph from p(z_ph)
	z_rnd = samples[np.random.randint(samples.shape[0]),0]

	del sampler

	# ID, RA, DEC, z_i, p(z_i), z_ph_max, z_rnd, f, s500/350/250, err_s500/350/250 
	return [ids, data[data['ID']==ids]['RA'], data[data['ID']==ids]['DEC'], _z_, p_z, z_ph_max, z_rnd, f, data[data['ID']==ids]['s500'][0], data[data['ID']==ids]['s350'][0], data[data['ID']==ids]['s250'][0], data[data['ID']==ids]['err_s500'][0], data[data['ID']==ids]['err_s350'][0], data[data['ID']==ids]['err_s250'][0]]


def write_table(tab, filename=None):
	# ID, RA, DEC, z_i, p(z_i), z_ph_max, z_rnd, f, s500/350/250, err_s500/350/250 
	col_id   = Column(data=[tab[i][0]  for i in xrange(len(tab))], name='ID')
	col_ra   = Column(data=[tab[i][1]  for i in xrange(len(tab))], name='RA')
	col_dec  = Column(data=[tab[i][2]  for i in xrange(len(tab))], name='DEC')
	col_zi   = Column(data=[tab[i][3]  for i in xrange(len(tab))], name='z_i', description='z values at which p(z) is sampled')
	col_pz   = Column(data=[tab[i][4]  for i in xrange(len(tab))], name='p_z', description='Posterior p(z)')
	col_zmax = Column(data=[tab[i][5]  for i in xrange(len(tab))], name='z_ph_max', description='Peak of posterior max[p(z)]')
	col_zrnd = Column(data=[tab[i][6]  for i in xrange(len(tab))], name='z_rnd', description='Random z drawn from posterior p(z)')
	col_f    = Column(data=[tab[i][7]  for i in xrange(len(tab))], name='f', description='Chain mean acceptance fraction')
	col_s500 = Column(data=[tab[i][8]  for i in xrange(len(tab))], name='s500', description='Flux @ 500 micron', unit='mJy')
	col_s350 = Column(data=[tab[i][9]  for i in xrange(len(tab))], name='s350', description='Flux @ 350 micron', unit='mJy')
	col_s250 = Column(data=[tab[i][10] for i in xrange(len(tab))], name='s250', description='Flux @ 250 micron', unit='mJy')
	col_e500 = Column(data=[tab[i][11] for i in xrange(len(tab))], name='err_s500', description='Flux Error @ 500 micron', unit='mJy')
	col_e350 = Column(data=[tab[i][12] for i in xrange(len(tab))], name='err_s350', description='Flux Error @ 350 micron', unit='mJy')
	col_e250 = Column(data=[tab[i][13] for i in xrange(len(tab))], name='err_s250', description='Flux Error @ 250 micron', unit='mJy')

	tot_tab = [col_id, col_ra, col_dec, col_zi, col_pz, col_zmax, col_zrnd, col_f, col_s500, col_s350, col_s250, col_e500, col_e350, col_e250]

	t = Table(tot_tab)
	if filename is None: 
		filename = 'new_catalog.fits'
	t.write(filename, format='fits')

# Let's start ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
	print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
	print '...Hello, let us get started...'

	# MCMC sampler parameters
	nsteps   = 500
	nwalkers = 40
	nburnin  = 200
	prior    = 'jeffreys'     # Prior Type
	p0       = [.5,1.]		  # Initial guess for z_ph and template amplitude A

	# Parameters 
	par_names = ['z_ph', 'A']
	par_label = ['$z_{ph}$', 'A']

	# Redshift array
	# _z_ = np.linspace(0,10,1000)

	# Interpolating the SED
	SED = get_SED_spline('SED_SMM_J2135.dat')
	print '...SED spline from SED_SMM_J2135.dat initialized...'

	# Loading objects specifics: ID, Fluxes, ...
	patches    = ['G09']#, 'G12', 'G15', 'NGP', 'SGP']
	catalogues = {'G09': 'hatlas_phase1_zsissa_G09_SMM_35mJy.dat',
				  'G12': 'hatlas_phase1_zsissa_G12_SMM_35mJy.dat',
				  'G15': 'hatlas_phase1_zsissa_G15_SMM_35mJy.dat',
				  'NGP': 'hatlas_NGP_psf_zsissa_SMM_35mJy.dat',
				  'SGP': 'hatlas_SGP_psf_zsissa_SMM_35mJy.dat'
				 }
	path_cats  = '/Users/federicobianchini/Documents/Universita/SISSA/XC_project/Codes/MASTER/herschel_cats/'

	names_cats = 'ID', 'RA', 'DEC', 's500', 's350', 's250', 'err_s500', 'err_s350', 'err_s250', 'zph', 'err_zph'

	for cat in patches:
		print '...analyzing catalouge ', cat,'...'
		data = np.genfromtxt(path_cats+catalogues[cat], skiprows=3, dtype=None, usecols=(0,1,2,3,4,5,8,9,10,13,14), names=names_cats)

		# Create workers processes and run parallel analysis 
		pool   = Pool(processes=2)
		result = pool.map(do_analysis, data['ID'][np.random.randint(10000,size=4)])
		# result = map(do_analysis, data['ID'][:6])
		# result = pool.map(do_analysis, data['ID'])
		pool.close()
		pool.join()
		del pool
		fname = cat+'_cat_35mJy.fits'
		write_table(result, filename=fname)
		print '... writing fits file ', fname,'...'
		print '...done catalouge ', cat,'...'

print '...GAME OVER...'
print '~~~~~~~~~~~~~~~'














