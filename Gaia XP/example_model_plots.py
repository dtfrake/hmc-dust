import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize, LogNorm
import matplotlib.gridspec as gridspec


class LogFormatterDecimal(ticker.LogFormatter):
     """
     Format values for log axis using decimal representation.
     """
     def _num_to_string(self, x, vmin, vmax):
         return f'{x:g}'


def plot_stellar_model(stellar_model, n_val=15):
    n_bands = 5
    sample_wavelengths = stellar_model._sample_wavelengths

    for param in ('teff', 'logg', 'feh', 'dwarfs', 'giants', 'E'):
        print(f"Plotting model flux vs {param}")

        teff = np.full(n_val, 5500.)
        logg = np.full(n_val, 4.6)
        feh = np.full(n_val, 0.)
        E = np.full(n_val, 0.)
        if param == 'dwarfs':
            teff = np.linspace(4000., 11000., n_val)
            logg = np.zeros_like(teff)
            logg[teff<5000] = 4.6
            idx = (teff>=5000) & (teff<6300)
            logg[idx] = 4.6-0.65*(teff[idx]-5000)/1300
            logg[teff>=6300] = 3.95
            label = r'$T_{\mathrm{eff}}$'
            title = (
                r'$'
                r'\mathrm{Main\ Sequence} ,\ '
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f}'
                '$'
            )
            v = teff
        elif param == 'giants':
            logg = np.linspace(4.15, 1.5, n_val)
            teff = np.zeros_like(logg)
            idx = (logg >= 3.65)
            teff[idx] = 5900 - 700*(4.15-logg[idx])/(4.15-3.65)
            teff[~idx] = 5200 - 950*(3.65-logg[~idx])/(3.65-1.5)
            label = r'$\log g$'
            title = (
                r'$'
                r'\mathrm{Giant\ Branch} ,\ '
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f}'
                '$'
            )
            v = logg
        elif param == 'teff':
            teff = np.linspace(4000., 6500., n_val)
            label = r'$T_{\mathrm{eff}}$'
            title = (
                r'$'
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f} ,\,'
                rf'\log g = {logg[0]:.1f}'
                '$'
            )
            v = teff
        elif param == 'logg':
            logg = np.linspace(2, 5., n_val)
            label = r'$\log g$'
            title = (
                r'$'
                rf'T_{{\mathrm{{eff}}}} = {teff[0]:.0f} ,\,'
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f}'
                '$'
            )
            v = logg
        elif param == 'feh':
            feh = np.linspace(-1.0, 0.4, n_val)
            label = r'$\left[\mathrm{Fe/H}\right]$'
            title = (
                r'$'
                rf'T_{{\mathrm{{eff}}}} = {teff[0]:.0f} ,\,'
                rf'\log g = {logg[0]:.1f}'
                '$'
            )
            v = feh
        elif param == 'E':
            E = np.linspace(0., 2, n_val)
            label = r'$E$'
            title = (
                r'$'
                rf'T_{{\mathrm{{eff}}}} = {teff[0]:.0f} ,\,'
                rf'\log g = {logg[0]:.1f},\,'
                rf'\left[\mathrm{{Fe/H}}\right] = {feh[0]:.1f}'
                '$'
            )
            v = E

        stellar_params = np.stack([0.001*teff,feh,logg], axis=1)
        flux = stellar_model.predict_observed_flux(stellar_params, E, np.ones(n_val))
        
        fig = plt.figure(figsize=(6,4))

        (gs,gs1) = gridspec.GridSpec(1, 2, width_ratios = [3, 1],  wspace = 0)
        ax = plt.subplot(gs)
        ax1 = plt.subplot(gs1)

        if param in ['feh', 'E']:
            cmap = cm.coolwarm
        else:
            cmap = cm.coolwarm_r
        norm = Normalize(vmin=min(v), vmax=max(v))

        ax_min,ax1_min,ax2_min = 10**20,10**20,10**20
        ax_max,ax1_max,ax2_max = 0,0,0
        
        for vv,fl in zip(v, flux):
            x = (sample_wavelengths[:-n_bands])
            y = fl[:-n_bands]*sample_wavelengths[:-n_bands]**2
            
            ax.loglog(
                x,
                y,
                color=cmap(norm(vv)),
                linewidth=0.5
            )
            
            ax_min = np.min([np.min(y),ax_min])
            ax_max = np.max([np.max(y),ax_max])
            
        for vv,fl in zip(v,flux):
            x = sample_wavelengths[-n_bands:]
            y = fl[-n_bands:]*sample_wavelengths[-n_bands:]**2
            
            ax1.scatter(
                x,
                y,
                color=cmap(norm(vv)),
                edgecolors='none',
                s=4
            )
            ax1_min = np.min([np.min(y),ax1_min])
            ax1_max = np.max([np.max(y),ax1_max])
        

        fig.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax1,
            label=label
        )

        ax.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')
        
        ylabel = (
            r'$\lambda^2 f_{\lambda}\ '
            r'\left(10^{-18}\,\mathrm{W\,m^{-2}\,nm}\right)$'
        )
        ax.set_ylabel(ylabel)

        plt.suptitle(title)
        
        plt.subplots_adjust(
            left=0.11,
            right=0.88,
            bottom=0.14,
            top=0.90,
            wspace=0.3
        )

        for a in (ax,ax1):
            a.tick_params(
                axis='both',
                direction='in',
                which='both'
            )
            
        ax.set_xlim([380,999.9])
        ax1.set_xscale('log')
        ax1.set_xlim([1000,6000])
        ax1.set_xticks([1000,4000],['1000','4000'])
        ax1.set_ylim(ax.get_ylim())
        
        ax1.set_yscale('log')
        ax1.yaxis.set_ticklabels(ticklabels=[])
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
             
        for a in (ax,ax1):
            a.grid(True, which='major', c='k', alpha=0.1)
        
        ax.grid(True, which='minor', c='k', alpha=0.04)         
        ax1.minorticks_off()
        
        fig.savefig(f'model_flux_vs_{param}.png')
        plt.close(fig)
        

def plot_ext_curve(stellar_model):
    n_bands = 5
    ln_ext_curve = stellar_model._ln_ext_curve
    ext_curve = np.exp(ln_ext_curve) * 2.5/np.log(10)
    wl = stellar_model._sample_wavelengths

    fig,ax = plt.subplots(1,1, figsize=(6,5))

    l, = ax.loglog(wl[:-n_bands], ext_curve[:-n_bands])
    ax.scatter(
        wl[-n_bands:],
        ext_curve[-n_bands:],
        color=l.get_color(), s=9
    )

    ax.grid(True, which='major', c='k', alpha=0.2)
    ax.grid(True, which='minor', c='k', alpha=0.05)

    ax.xaxis.set_major_formatter(
        LogFormatterDecimal(minor_thresholds=(3,0.4))
    )
    ax.xaxis.set_minor_formatter(
        LogFormatterDecimal(minor_thresholds=(3,0.4))
    )
    ax.yaxis.set_major_formatter(
        LogFormatterDecimal(minor_thresholds=(3,0.4))
    )
    ax.yaxis.set_minor_formatter(
        LogFormatterDecimal(minor_thresholds=(3,0.4))
    )

    ax.set_xlabel(r'$\lambda\ \left(\mathrm{nm}\right)$')
    ax.set_ylabel(r'$A\left(\lambda\right)$')
    ax.set_title(r'$\mathrm{Extinction\ curve}$')

    fig.subplots_adjust(
        left=0.11,
        right=0.96,
        bottom=0.12,
        top=0.92
    )
    ax.tick_params(axis='both', labelsize=10, 
                   direction='in', width=1, length=4, which='both')

    fig.savefig('extinction_curve.png')
    plt.close(fig)


def main():
    # Load the model
    stellar_model = tf.saved_model.load('stellar_flux_model')
    # Plot model flux vs param
    plot_stellar_model(stellar_model, n_val=15)
    # Plot extinction curve vs wavelength
    plot_ext_curve(stellar_model)


if __name__ == "__main__":
    main()
