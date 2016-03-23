import numpy as np
from scipy import interpolate, integrate
import matplotlib.pyplot as plt
import healpy as hp
# from IPython import embed
from astropy.table import Table, Column, vstack

def get_counts_map(ra, dec, nside, galactic=True, nest=False):
	"""
	Creates an Healpix map with galaxy number counts at resolution nside given two arrays containing
	RA and DEC positions 
	"""
	theta = np.deg2rad(90.-dec) 
	phi   = np.deg2rad(ra)      

	# Apply rotation from EQ -> GAL if needed
	if galactic:
		r = hp.Rotator(coord=['C','G'], deg=False)
		theta, phi = r(theta, phi)

	# Converting galaxy coordinates -> pixel 
	npix = hp.nside2npix(nside)
	pix  = hp.ang2pix(nside, theta, phi, nest=nest)

	# Create Healpix map
	counts_map = np.bincount(pix, minlength=npix)

	return counts_map

def get_pz_spline(pz, zi, kind='slinear', fill_value=0.):
	"""
	Returns interpolation object given the sampled p(z) at values zi.
	"""
	return interpolate.interp1d(zi, pz, kind='slinear', fill_value=fill_value, bounds_error=False)

def get_pz(table, norm=True, zmin=0., zmax=10.):
	"""
	Returns the stacked p(z) for *all* the objects in table
	"""
	z_ = np.linspace(zmin, zmax, 1000)
	pz = np.zeros(1000)
	for row in table:
		pz += get_pz_spline(row['p_z'], row['z_i'])(z_)
	if norm:
		pz /= integrate.simps(pz, x=z_)

	return pz

def zbin_table(table, field, zbins):
    """
    Returns a dictionary with Table objects containing sources in given zbin (specified in zbins dict)
    table: astropy table object
    field: astropy table object field (either z_ph_max or z_rnd)
    zbins: dictionary containing (zbin_name: (z_bin_min, z_bin_max)) as {'1.5_10':(1.5,10), '1.0_4':(1.,4.)}
    
    tables: dictionary of tables 
    """
    col    = table[field]
    masks  = ((col >= inf) & (col < sup) for (inf, sup) in zbins.values())
    tabs   = [table[mask] for mask in masks]
    tables = dict(zip(zbins.keys(), tabs))
    return tables

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

def dndz2phi(dNdz, z, zbins=(0., 10.), sigma=0.26, bias=0, nbins=30):
	dNdz_cat = get_dNdz_spline(dNdz, z)
	z_       = np.linspace(0., 10, 1000)
	phi      = dNdz_cat(z_) * convolve_window(z_, sigma=sigma, bias=bias, z_min=zbins[0], z_max=zbins[1])
	phi_norm = integrate.simps(phi, x=z_)
	phi     /= phi_norm

	return interpolate.interp1d(z_, phi, bounds_error=False, fill_value=0.)



if __name__ == '__main__':

	patches = ['G09', 'G12', 'G15', 'NGP', 'SGP', 'ALL']

	# Acceptance fraction threshold
	f_min = 0.1 

	# Redshift bins
	z_bins = {'1.5_10':(1.5, 10.), '1.5_2.1':(1.5, 2.1), '2.1_10':(2.1, 10.)}
	# z_bins = None

	# Nside map resolution
	nside = 512

	# Redshift array
	z_ = np.linspace(0.,10.,1000)

	# Maps: gal_counts_tot[patch], gal_counts_xxxx[patch][zbin]
	gal_counts_tot = {}
	if z_bins is not None:
		gal_counts_zmax = {} # Dictionary containing maps obtained with photo-z as z_ph_max
		gal_counts_zrnd = {} # Dictionary containing maps obtained with photo-z as z_rnd
		for patch in patches:
			gal_counts_zmax[patch] = {}
			gal_counts_zrnd[patch] = {}

	# p(z): p_z_tot[patch], p_z_xxx[patch][zbin]
	p_z_tot = {} # Dictionary containing p(z) obtained stacking all photo-z posterior
	if z_bins is not None:
		p_z_max = {} # Dictionary containing stacked photo-z posterior of z_ph_max p(z) (in the z-bin)
		p_z_rnd = {} # Dictionary containing stacked photo-z posterior of z_rnd    p(z) (in the z-bin)
		p_z_con = {} # Dictionary containing convolved *full* photo-z posterior p(z) (in the z-bin) [a la Budavari]
		for patch in patches:
			p_z_max[patch] = {}
			p_z_rnd[patch] = {}
			p_z_con[patch] = {}

	# Doin' analysis over patches
	for patch in patches:
		print '...analyzing catalouge ', patch,'...'
		
		# Reading Table file and discarding objects with acceptance fraction < f_min
		if patch == 'ALL': # FIXME!!!!!
			tab_ = {}
			for p in ['G09', 'G12', 'G15', 'NGP', 'SGP']:
				fits_file = p+'_cat_35mJy_correct.fits'
				tab_[p] = Table.read(fits_file, format='fits')
				tab_[p] = tab_[p][tab_[p]['f']>f_min]  
			tab = vstack([tab_['G09'], tab_['G12'], tab_['G15'], tab_['NGP'], tab_['SGP']])
		else:
			fits_file = patch+'_cat_35mJy_correct.fits'
			tab = Table.read(fits_file, format='fits')
			tab = tab[tab['f']>f_min]  
		
		# Evaluating total p(z)
		print '   evaluating total p(z)...'
		p_z_tot[patch] = get_pz(tab)
		np.savetxt('pz/pz_stacked_only35_'+patch+'_tot.dat', np.c_[z_, p_z_tot[patch]])
		print '   total p(z) evaluated...'

		# Histograms vs. p(z) plot (no z-cut)
		if True:
			plt.title(patch)
			plt.hist(tab['z_ph_max'], 30, label=r'$z^{max}_{ph}$', normed=True, histtype='step')
			plt.hist(tab['z_rnd'], 30,    label=r'$z^{rnd}_{ph}$', normed=True, histtype='step')
			plt.plot(z_, p_z_tot[patch],  label=r'Stacked $p(z)$ - no cuts')
			plt.legend()
			plt.xlabel(r'$z$')
			plt.ylabel(r'$p(z)$ [arbitrary units]')
			plt.xlim([0,4.])
			plt.savefig('plots/hist_zmax_zrnd_pz_stack_'+patch+'.pdf', bbox_inches='tight')
			plt.close()
			print '   histogram plot done...'

		# Evaluating p(z) for z_ph_max and z_rnd methods (if zbin is not None)
		if z_bins is not None:
			print '   evaluating p(z) in z-bins...'
			tab_zmax = zbin_table(tab, 'z_ph_max', z_bins) 
			tab_zrnd = zbin_table(tab, 'z_rnd',    z_bins)
			for zbin, zbin_edges in z_bins.iteritems():
				print '   --> ',str(zbin_edges)
				p_z_max[patch][zbin] = get_pz(tab_zmax[zbin])
				p_z_rnd[patch][zbin] = get_pz(tab_zrnd[zbin])
				p_z_con[patch][zbin] = dndz2phi(p_z_tot[patch], z_, zbins=zbin_edges, sigma=0.26, bias=0)(z_)

				# Save p(z)
				np.savetxt('pz/pz_stacked_only35_'+str(zbin_edges[0])+'_'+str(zbin_edges[1])+'_'+patch+'_zphmax.dat', np.c_[z_, p_z_max[patch][zbin]])
				np.savetxt('pz/pz_stacked_only35_'+str(zbin_edges[0])+'_'+str(zbin_edges[1])+'_'+patch+'_zrnd.dat',   np.c_[z_, p_z_rnd[patch][zbin]])
				np.savetxt('pz/pz_convolved_only35_'+str(zbin_edges[0])+'_'+str(zbin_edges[1])+'_'+patch+'_zcon.dat', np.c_[z_, p_z_con[patch][zbin]])

				# Different p(z) plots in zbin
				if True:
					plt.title(patch+''+zbin)
					plt.plot(z_, p_z_max[patch][zbin],  label=r'Stacked $p^{max}(z)$')
					plt.plot(z_, p_z_rnd[patch][zbin],  label=r'Stacked $p^{rnd}(z)$')
					plt.plot(z_, p_z_con[patch][zbin],  label=r'Convolved $p^{con}(z)$')
					plt.axvspan(zbin_edges[0], zbin_edges[1], alpha=0.4, color='grey', label='$W(z_{ph})$')
					plt.legend()
					plt.xlabel(r'$z$')
					plt.ylabel(r'$p(z)$ [arbitrary units]')
					plt.xlim([0,4.])
					plt.savefig('plots/pz_zmax_zrnd_stack_conv_'+str(zbin_edges[0])+'_'+str(zbin_edges[1])+'_'+patch+'.pdf', bbox_inches='tight')
					plt.close()
					print '      z-bins plot done...'

		# Maps creation 
		print '   creating Healpix galaxy number counts maps...'
		gal_counts_tot[patch] = get_counts_map(tab['RA'], tab['DEC'], nside)
		hp.write_map('maps/herschel_counts_nozcut_'+str(patch)+'_gal_'+str(nside)+'_SMM_35mJy_photoz.fits', gal_counts_tot[patch])
		print '   total map done...'
		if z_bins is not None:
			print '   creating Healpix galaxy number counts maps in z-bins...'
			for zbin, zbin_edges in z_bins.iteritems():
				print '   --> ',str(zbin_edges)
				gal_counts_zmax[patch][zbin] = get_counts_map(tab_zmax[zbin]['RA'], tab_zmax[zbin]['DEC'], nside)
				gal_counts_zrnd[patch][zbin] = get_counts_map(tab_zrnd[zbin]['RA'], tab_zrnd[zbin]['DEC'], nside)
				
				# Save maps
				hp.write_map('maps/herschel_counts_zcut_'+str(zbin_edges[0])+'_'+str(zbin_edges[1])+'_'+str(patch)+'_gal_'+str(nside)+'_SMM_35mJy_photoz_zphmax.fits', gal_counts_zmax[patch][zbin])
				hp.write_map('maps/herschel_counts_zcut_'+str(zbin_edges[0])+'_'+str(zbin_edges[1])+'_'+str(patch)+'_gal_'+str(nside)+'_SMM_35mJy_photoz_zrnd.fits',   gal_counts_zrnd[patch][zbin])
				print '       maps saved...'
		
		# Free memory
		del tab, tab_zrnd, tab_zmax

	print '...GAME OVER...'






