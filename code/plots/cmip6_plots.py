"""
TODO: module docstring

TODO: add reference(s): Marzeion (20202)

"""

# import build ins
import os
import logging

# import external libraries
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D

# instance logger
log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=getattr(logging, 'INFO'))

# define paths (templates)
frac_csv_tpl = '/home/www/moberrauch/merge_cru_cmip_{}/frac_output/{}/{}.csv'
frac_csv_glob = '/home/www/moberrauch/merge_cru_cmip_{}/frac_output/global.csv'
abs_csv_tpl = '/home/www/moberrauch/merge_cru_cmip_{}/abs_output/{}/{}.csv'
abs_csv_glob = '/home/www/moberrauch/merge_cru_cmip_{}/abs_output/global.csv'
plots_parent_dir = '/home/users/moberrauch/paper/plots/'
data_dir = '/home/users/moberrauch/paper/data/'


def cmap_to_clist(cmap_name, n):
    """Convert colormap to a list of `n` discrete colors.

    Parameters
    ----------
    cmap_name : str
        Name of colormap
    n : int
        Number of colors, i.e. length of returned list

    Returns
    -------
    list : list of discrete colors (str, hex format)

    """
    cmap = plt.cm.get_cmap(cmap_name)
    return [mcolors.to_hex(c) for c in cmap(np.linspace(0, 1, int(n)))]


def make_patch_spines_invisible(ax):
    """TODO: if not in use delete."""
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def compute_slr(ice_vol_m3):
    """Convert ice volume from cubic meters (m3) into millimeters sea level
    equivalent (mm SLE)"""
    rho_ice = 900  # ice density [kg/m3]
    rho_ocean = 1028  # density of ocean water [kg/m3]
    area_ocean = 362.5 * 1e12  # ocean surface area [m2]
    slr_mm = ice_vol_m3 * rho_ice / (area_ocean * rho_ocean) * 1e3
    return slr_mm


def rainbow_text(x, y, strings, colors, orientation='horizontal',
                 ax=None, **kwargs):
    """
    Take a list of *strings* and *colors* and place them next to each
    other, with text strings[i] being shown in colors[i].

    Parameters
    ----------
    x, y : float
        Text position in data coordinates.
    strings : list of str
        The strings to draw.
    colors : list of color
        The colors to use.
    orientation : {'horizontal', 'vertical'}
    ax : Axes, optional
        The Axes to draw into. If None, the current axes will be used.
    **kwargs
        All other keyword arguments are passed to plt.text(), so you can
        set the font size, family, etc.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    assert orientation in ['horizontal', 'vertical']
    if orientation == 'vertical':
        kwargs.update(rotation=90, verticalalignment='bottom')

    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kwargs)

        # Need to draw to update the text position.
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        if orientation == 'horizontal':
            t = text.get_transform() + Affine2D().translate(ex.width, 0)
        else:
            t = text.get_transform() + Affine2D().translate(0, ex.height)


def plot_regional_mass_loss_rel(ref_year=2020,
                                figsize=None, n_columns=3,
                                ssps=('ssp126', 'ssp245', 'ssp370', 'ssp585'),
                                save_fig=True, secondary_axis=False):
    """Plot regional projections of both models for the same SSP(s) with
    relative mass loss as main (left) axis, in analogy to Figure 6 in
    [Marzeion 2020]_

    Each combination of RGI region and SSP results in one plot, containing the
    projections (average +/- one standard deviation of all CMIP models) of the
    VAS model and the flowline model. Thereby, the shared (left) axis shows the
    fraction of the reference volume (volume at the reference year) in
    percentage. The two right axes show the absolute values of ice volume (in
    mm SLE) separately for each model, again in relation to the reference year.
    """
    plt.style.use('./paper.mplstyle')

    # read rgi region file (sorted with descending 2020 ice volume)
    rgi_reg_df = pd.read_csv(os.path.join(data_dir, 'rgi_reg_df.csv'),
                             index_col=0)

    # iterate over all given shared socioeconomic pathways
    ssps = np.atleast_1d(ssps)
    for ssp in ssps:
        log.info(f'Creating plots for SSP{ssp[-3:]}')
        # iterate over all rgi regions
        for i, (rgi_reg, row) in enumerate(rgi_reg_df.iterrows()):
            log.info(f'Creating plots for RGI region {rgi_reg}')
            # convert region number from integer to zero leading string
            rgi_reg = f'{rgi_reg:02d}'

            # define colors
            colors = np.array(["#e63946", "#457b9d"])

            # plot y-axis label only on the first plot of each row
            ylabel = not (i % n_columns)

            # create figure with given figure size
            if secondary_axis:
                if not figsize:
                    figsize = (6, 4.5)
                fig = plt.figure(figsize=figsize)
                # frac_ax = fig.add_axes([0.1, 0.1, 0.65, 0.85])
                frac_ax = fig.add_axes([0.1, 0.1, 0.65, 0.85])
            else:
                if not figsize:
                    if ylabel:
                        figsize = (6.7, 4.5)
                        ax_rect = [0.165, 0.10, 0.78, 0.8]
                    else:
                        figsize = (6.5, 4.5)
                        ax_rect = [0.13, 0.10, 0.8, 0.8]
                else:
                    ax_rect = [0.15, 0.1, 0.8, 0.85]
                fig = plt.figure(figsize=figsize)
                frac_ax = fig.add_axes(ax_rect)

            # plot projections for both models
            for c, model in zip(colors, ['vas', 'fl']):
                # read relative ice volume records
                frac = pd.read_csv(frac_csv_tpl.format(model, rgi_reg, ssp),
                                   index_col=0)
                # compute average and standard deviation
                avg = frac.mean(axis=1).loc[2000:2100]
                std = frac.std(axis=1).loc[2000:2100]

                # plot one std deviation range
                frac_ax.fill_between(avg.index, avg - std, avg + std,
                                     alpha=0.5,
                                     color=c)
                # define label
                model_label = 'VAS' if model == 'vas' else 'Flowline'
                label = model_label + '  ({}%)'.format(
                    int((avg.loc[2100] - 1) * 100))
                # plot the ensemble average
                avg.plot(label=label, color=c, lw=3, ax=frac_ax)

            # add labels, legend, etc.
            plt.xlabel('')
            if ylabel:
                plt.ylabel(f'Faction of {ref_year} volume')
            else:
                plt.ylabel('')
            plt.legend(loc='lower left')
            plt.ylim([0, 1.4])
            plt.xlim(2000, 2100)
            plt.axhline(1, c='grey', ls=':', lw=0.8)
            plt.grid()

            # add region as text
            frac_ax.text(1, 1.01, row.region_name,
                         transform=frac_ax.transAxes, ha='right')

            if secondary_axis:
                # get limits of current y-axis showing the volume fraction
                mn, mx = frac_ax.get_ylim()

                # secondary y-axis with absolute values (mm SLE) for VAS model
                mm_sle_ax_vas = frac_ax.twinx()
                abs_vol = pd.read_csv(abs_csv_tpl.format('vas', rgi_reg, ssp),
                                      index_col=0)
                sle_2020 = compute_slr(abs_vol.loc[2020].iloc[0])
                mm_sle_ax_vas.set_ylim((1 - np.array([mn, mx])) * sle_2020)
                # limit number of ticks/labels to 5 (and thereby number of decimal
                # points to 0.00)
                mm_sle_ax_vas.yaxis.set_major_locator(mticker.MaxNLocator(5))
                mm_sle_ax_vas.set_ylabel(
                    'Sea level rise for the VAS model (mm)')

                # secondary y-axis with absolute values (mm SLE) for flowline model
                mm_sle_ax_fl = frac_ax.twinx()
                abs_vol = pd.read_csv(abs_csv_tpl.format('fl', rgi_reg, ssp),
                                      index_col=0)
                sle_2020 = compute_slr(abs_vol.loc[2020].iloc[0])
                mm_sle_ax_fl.set_ylim((1 - np.array([mn, mx])) * sle_2020)
                # limit number of ticks/labels to 5 (and thereby number of decimal
                # points to 0.00)
                mm_sle_ax_fl.yaxis.set_major_locator(mticker.MaxNLocator(5))
                mm_sle_ax_fl.set_ylabel(
                    'Sea level rise for the flowline model (mm)')
                mm_sle_ax_fl.spines["right"].set_position(("axes", +1.2))

            # save to file
            if save_fig:
                fig_path = os.path.join(plots_parent_dir, 'cmip_single_ssp',
                                        'frac_vol', 'regional',
                                        f'cmip_{ssp}_{rgi_reg}.pdf')
                fig.savefig(fig_path)


def plot_global_mass_loss_rel(ref_year=2020,
                              figsize=(6, 4.5),
                              ssps=('ssp126', 'ssp245', 'ssp370', 'ssp585'),
                              save_fig=True, secondary_axis=True):
    """Plot global projections of both models for the same SSP(s)

    One plot for each SSP, containing the projections (average +/- one standard
    deviation of all CMIP models) of the VAS model and the flowline model.
    Thereby, the shared (left) axis shows the fraction of the reference volume
    (volume at the reference year) in percentage. The two right axes show the
    absolute values of ice volume (in mm SLE) separately for each model, again
    in relation to the reference year.
    """
    # iterate over all given shared socioeconomic pathways
    for ssp in ssps:
        log.info(f'Creating plots for SSP{ssp[-3:]}')

        # define colors
        colors = np.array(["#e63946", "#457b9d"])

        # create figure with given figure size
        fig = plt.figure(figsize=figsize)
        frac_ax = fig.add_axes([0.1, 0.1, 0.65, 0.85])

        # plot projections for both models
        for c, model in zip(colors, ['vas', 'fl']):
            # read relative ice volume records
            frac = pd.read_csv(frac_csv_glob.format(model), index_col=0)
            # subset for given SSP
            frac = frac.loc[:, [ssp in column for column in frac.columns]]
            # compute average and standard deviation
            avg = frac.mean(axis=1).loc[2000:2100]
            std = frac.std(axis=1).loc[2000:2100]

            # plot one std deviation range
            frac_ax.fill_between(avg.index, avg - std, avg + std, alpha=0.3,
                                 color=c)
            # define label
            model_label = 'VAS' if model == 'vas' else 'Flowline'
            label = model_label + '  ({}%)'.format(
                int((avg.loc[2100] - 1) * 100))
            # plot the ensemble average
            avg.plot(label=label, color=c, lw=2, ax=frac_ax)

        # add labels, legend, etc.
        plt.xlabel('')
        plt.ylabel(f'Faction of {ref_year} volume')
        plt.legend(loc='lower left')
        plt.xlim([2000, 2100])
        plt.ylim([0, 1.1])
        plt.axhline(1, c='grey', ls=':', lw=0.8)
        plt.grid()

        # add SSP as text
        frac_ax.text(1, 1.01, f'SSP{ssp[3]}-{float(ssp[4:])/10:.1f}',
                     transform=frac_ax.transAxes, ha='right')

        if secondary_axis:
            # get limits of current y-axis showing the volume fraction
            mn, mx = frac_ax.get_ylim()

            # secondary y-axis with absolute values (mm SLE) for VAS model
            mm_sle_ax_vas = frac_ax.twinx()
            abs_vol = pd.read_csv(abs_csv_glob.format('vas'), index_col=0)
            sle_2020 = compute_slr(abs_vol.loc[2020].iloc[0])
            mm_sle_ax_vas.set_ylim((1 - np.array([mn, mx])) * sle_2020)
            mm_sle_ax_vas.set_ylabel('Sea level rise for the VAS model (mm)')
            # secondary y-axis with absolute values (mm SLE) for flowline model
            mm_sle_ax_fl = frac_ax.twinx()
            abs_vol = pd.read_csv(abs_csv_glob.format('fl'), index_col=0)
            sle_2020 = compute_slr(abs_vol.loc[2020].iloc[0])
            mm_sle_ax_fl.set_ylim((1 - np.array([mn, mx])) * sle_2020)
            mm_sle_ax_fl.set_ylabel(
                'Sea level rise for the flowline model (mm)')
            mm_sle_ax_fl.spines["right"].set_position(("axes", +1.2))

        # save to file
        if save_fig:
            fig_path = os.path.join(plots_parent_dir, 'cmip_single_ssp',
                                    'frac_vol', 'global',
                                    f'cmip_{ssp}_global.pdf')
            fig.savefig(fig_path)


def plot_regional_mass_loss_abs(rgi_regions=np.arange(1, 19),
                                ref_year=2020,
                                figsize=(6, 4.5),
                                ssps=('ssp126', 'ssp245', 'ssp370', 'ssp585'),
                                save_fig=True):
    """Plot regional projections of both models for the same SSP(s)

    Each combination of RGI region and SSP results in one plot, containing the
    projections (average +/- one standard deviation of all CMIP models) of the
    VAS model and the flowline model. Thereby, the shared (left) axis shows the
    fraction of the reference volume (volume at the reference year) in
    percentage. The two right axes show the absolute values of ice volume (in
    mm SLE) separately for each model, again in relation to the reference year.
    """
    # iterate over all given shared socioeconomic pathways
    for ssp in ssps:
        log.info(f'Creating plots for SSP{ssp[-3:]}')
        # iterate over all rgi regions
        for rgi_reg in rgi_regions:
            log.info(f'Creating plots for RGI region {rgi_reg}')

            # convert region number from integer to zero leading string
            rgi_reg = f'{rgi_reg:02d}'

            # define colors
            colors = np.array(["#e63946", "#457b9d"])

            # create empty container to store reference volume
            # used later for secondary axis
            ref_slr = dict()

            # create figure with given figure size
            fig = plt.figure(figsize=figsize)
            mm_sle_ax = fig.add_axes([0.1, 0.1, 0.65, 0.85])

            # plot projections for both models
            for c, model in zip(colors, ['vas', 'fl']):
                # read relative ice volume records
                abs_vol = pd.read_csv(abs_csv_tpl.format(model, rgi_reg, ssp),
                                      index_col=0)
                # get reference volume (mm SLE) and store for later use
                ref_vol = abs_vol.loc[ref_year]
                ref_slr[model] = compute_slr(ref_vol[0])
                # compute sea level rise contribution relative to ref. year
                mm_slr = compute_slr(ref_vol - abs_vol)
                # compute average and standard deviation
                avg = mm_slr.mean(axis=1).loc[2000:2100]
                std = mm_slr.std(axis=1).loc[2000:2100]

                # plot one std deviation range
                mm_sle_ax.fill_between(avg.index, avg - std, avg + std,
                                       alpha=0.3,
                                       color=c)
                # define label
                model_label = 'VAS' if model == 'vas' else 'Flowline'
                label = model_label + '  ({}%)'.format(
                    int((avg.loc[2100] - 1) * 100))
                # plot the ensemble average
                avg.plot(label=label, color=c, lw=2, ax=mm_sle_ax)

            # add labels, legend, etc.
            plt.xlabel('')
            plt.ylabel(
                f'Sea level rise contribution (mm) relative to {ref_year}')
            plt.legend(loc='upper left')
            plt.xlim(2000, 2100)
            plt.axhline(0, c='grey', ls=':', lw=0.8)
            plt.grid()

            # add region as text
            mm_sle_ax.text(1, 1.01, f'RGI region {rgi_reg}',
                           transform=mm_sle_ax.transAxes, ha='right')

            # get limits of current y-axis showing the volume fraction
            mm_sle_ax.invert_yaxis()
            mn, mx = mm_sle_ax.get_ylim()

            # add secondary y-axis with absolute values for VAS model
            frac_ax_vas = mm_sle_ax.twinx()
            frac_ax_vas.set_ylim((np.array([mn, mx])) / ref_slr['vas'] * 100)
            frac_ax_vas.set_ylabel(
                'Relative mass change (%) for the VAS model')
            # add secondary y-axis with absolute values for flowline model
            frac_ax_fl = mm_sle_ax.twinx()
            frac_ax_fl.set_ylim((np.array([mn, mx])) / ref_slr['fl'] * 100)
            frac_ax_fl.set_ylabel(
                'Relative mass change (%) for the flowline model')
            frac_ax_fl.spines["right"].set_position(("axes", +1.2))

            # save to file
            if save_fig:
                fig_path = os.path.join(plots_parent_dir, 'cmip_single_ssp',
                                        'abs_vol', 'regional',
                                        f'cmip_{ssp}_{rgi_reg}.pdf')
                fig.savefig(fig_path)


def plot_global_mass_loss_abs(ref_year=2020,
                              figsize=(6, 4.5),
                              ssps=('ssp126', 'ssp245', 'ssp370', 'ssp585'),
                              save_fig=True):
    """Plot global projections of both models for the same SSP(s)

    TODO: finish/adapt docstring

    One plot for each SSP, containing the projections (average +/- one standard
    deviation of all CMIP models) of the VAS model and the flowline model.
    Thereby, the shared (left) axis shows the fraction of the reference volume
    (volume at the reference year) in percentage. The two right axes show the
    absolute values of ice volume (in mm SLE) separately for each model, again
    in relation to the reference year.
    """
    # iterate over all given shared socioeconomic pathways
    for ssp in ssps:
        log.info(f'Creating plots for SSP{ssp[-3:]}')

        # define colors
        colors = np.array(["#e63946", "#457b9d"])

        # create empty container to store reference volume
        # used later for secondary axis
        ref_slr = dict()

        # create figure with given figure size
        fig = plt.figure(figsize=figsize)
        mm_sle_ax = fig.add_axes([0.1, 0.1, 0.65, 0.85])

        # plot projections for both models
        for c, model in zip(colors, ['vas', 'fl']):
            # read relative ice volume records
            abs_vol = pd.read_csv(abs_csv_glob.format(model),
                                  index_col=0)
            # subset for given SSP
            abs_vol = abs_vol.loc[:,
                      [ssp in column for column in abs_vol.columns]]
            # get reference volume (mm SLE) and store for later use
            ref_vol = abs_vol.loc[ref_year]
            ref_slr[model] = compute_slr(ref_vol[0])
            # compute sea level rise contribution relative to ref. year
            mm_slr = compute_slr(ref_vol - abs_vol)
            # compute average and standard deviation
            avg = mm_slr.mean(axis=1).loc[2000:2100]
            std = mm_slr.std(axis=1).loc[2000:2100]

            # plot one std deviation range
            mm_sle_ax.fill_between(avg.index, avg - std, avg + std, alpha=0.3,
                                   color=c)
            # define label
            model_label = 'VAS' if model == 'vas' else 'Flowline'
            label = model_label + '  ({:.0f} mm)'.format(avg.loc[2100])
            # plot the ensemble average
            avg.plot(label=label, color=c, lw=2, ax=mm_sle_ax)

        # add labels, legend, etc.
        plt.xlabel('')
        plt.ylabel(
            f'Sea level rise contribution (mm) relative to {ref_year}')
        plt.legend(loc='lower left')
        plt.xlim(2000, 2100)
        ylim = plt.gca().get_ylim()
        plt.ylim((ylim[0], max(ref_slr.values())))
        plt.axhline(0, c='grey', ls=':', lw=0.8)
        plt.grid()

        # add SSP as text
        mm_sle_ax.text(1, 1.01, f'SSP{ssp[3]}-{float(ssp[4:])/10:.1f}',
                       transform=mm_sle_ax.transAxes, ha='right')

        # get current y-axis (and limits) showing the absolute volume
        mm_sle_ax.invert_yaxis()
        mn, mx = mm_sle_ax.get_ylim()

        # add secondary y-axis with absolute values for VAS model
        frac_ax_vas = mm_sle_ax.twinx()
        frac_ax_vas.set_ylim((np.array([mn, mx])) / ref_slr['vas'] * 100)
        frac_ax_vas.set_ylabel(
            'Relative mass change (%) for the VAS model')
        # add secondary y-axis with absolute values for flowline model
        frac_ax_fl = mm_sle_ax.twinx()
        frac_ax_fl.set_ylim((np.array([mn, mx])) / ref_slr['fl'] * 100)
        frac_ax_fl.set_ylabel(
            'Relative mass change (%) for the flowline model')
        frac_ax_fl.spines["right"].set_position(("axes", +1.2))

        # save to file
        if save_fig:
            fig_path = os.path.join(plots_parent_dir, 'cmip_single_ssp',
                                    'abs_vol', 'global',
                                    f'cmip_{ssp}_global.pdf')
            fig.savefig(fig_path)


def plot_global_mass_loss_rel_all(ref_year=2020,
                                  save_fig=True, secondary_axis=True):
    """Plot global projections of both models for the same SSP(s)

    One plot for each SSP, containing the projections (average +/- one standard
    deviation of all CMIP models) of the VAS model and the flowline model.
    Thereby, the shared (left) axis shows the fraction of the reference volume
    (volume at the reference year) in percentage. The two right axes show the
    absolute values of ice volume (in mm SLE) separately for each model, again
    in relation to the reference year.
    """
    # create figure with given figure size
    plt.style.use('./igs.mplstyle')
    fig = plt.figure(figsize=(6.929, 6))

    # specify SSPs
    ssps = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
    n_tot = len(ssps)
    n_columns = 2

    # iterate over all given shared socioeconomic pathways
    for i, ssp in enumerate(ssps):
        log.info(f'Creating plots for SSP{ssp[-3:]}')

        # define colors
        colors = np.array(["#e63946", "#457b9d"])

        # create axes
        frac_ax = fig.add_subplot(int(np.ceil(n_tot / n_columns)),
                                  n_columns,
                                  i + 1)

        ylabel = not (i % n_columns)
        xlabel = i >= (n_tot - n_columns)

        text = list()
        text_tpl = '{} model: {:.0f}±{:.0f} mm SLE ({}±{}%)'

        # plot projections for both models
        for c, model in zip(colors, ['vas', 'fl']):
            # read relative ice volume records
            frac = pd.read_csv(frac_csv_glob.format(model), index_col=0)
            # subset for given SSP
            frac = frac.loc[:, [ssp in column for column in frac.columns]]
            # compute average and standard deviation
            avg_frac = frac.mean(axis=1).loc[2000:2100]
            std_frac = frac.std(axis=1).loc[2000:2100]

            # plot one std deviation range
            frac_ax.fill_between(avg_frac.index, avg_frac - std_frac,
                                 avg_frac + std_frac, alpha=0.3,
                                 color=c)
            # define label
            model_label = 'VAS' if model == 'vas' else 'Flowline'
            label = model_label + '  ({}%)'.format(
                int((avg_frac.loc[2100] - 1) * 100))

            # read absolute volume
            abs_vol = pd.read_csv(abs_csv_glob.format(model), index_col=0)
            # subset for given SSP
            abs_vol = abs_vol.loc[:,
                      [ssp in column for column in abs_vol.columns]]
            # compute average and standard deviation
            sle_2020 = compute_slr(abs_vol.loc[2020].iloc[0])
            avg_abs = sle_2020 - compute_slr(abs_vol.mean(axis=1).loc[2100])
            std_abs = compute_slr(abs_vol.std(axis=1).loc[2100])
            # define and store text
            model_name = 'Scaling' if model == 'vas' else 'Flowline'
            text.append(text_tpl.format(model_name,
                                        avg_abs, std_abs,
                                        int((1 - avg_frac.loc[2100]) * 100),
                                        int(std_frac.loc[2100] * 100)))

            # plot the ensemble average
            avg_frac.plot(label=label, color=c, lw=2, ax=frac_ax)

        # add labels, legend, etc.
        if not xlabel:
            frac_ax.get_xaxis().set_ticklabels([])
        frac_ax.set_xlabel('')
        if ylabel:
            frac_ax.set_ylabel(f'M (rel. to {ref_year})')
        else:
            frac_ax.get_yaxis().set_ticklabels([])
            frac_ax.set_ylabel('')

        frac_ax.text(2050, 0.2, text[0], color=colors[0], ha='center')
        frac_ax.text(2050, 0.1, text[1], color=colors[1], ha='center')

        # plt.legend(loc='lower left', fancybox=False)
        plt.xlim([2000, 2100])
        plt.ylim([0, 1.1])
        plt.axhline(1, c='grey', ls=':', lw=0.8)
        # plt.grid()

        # add SSP as text
        frac_ax.text(1, 1.01, f'SSP{ssp[3]}-{float(ssp[4:])/10:.1f}',
                     transform=frac_ax.transAxes, ha='right')

        if secondary_axis and not ylabel:
            # get limits of current y-axis showing the volume fraction
            mn, mx = frac_ax.get_ylim()

            # secondary y-axis with absolute values (mm SLE) for VAS model
            mm_sle_ax_vas = frac_ax.twinx()
            abs_vol = pd.read_csv(abs_csv_glob.format('vas'), index_col=0)
            sle_2020 = compute_slr(abs_vol.loc[2020].iloc[0])
            mm_sle_ax_vas.set_ylim((1 - np.array([mn, mx])) * sle_2020)
            mm_sle_ax_vas.set_ylabel('$\Delta{}$M (mm SLE) – scaling model',
                                     color=colors[0])
            mm_sle_ax_vas.spines["right"].set_color(colors[0])
            mm_sle_ax_vas.tick_params(axis='y', direction='inout',
                                      colors=colors[0])
            # secondary y-axis with absolute values (mm SLE) for flowline model
            mm_sle_ax_fl = frac_ax.twinx()
            abs_vol = pd.read_csv(abs_csv_glob.format('fl'), index_col=0)
            sle_2020 = compute_slr(abs_vol.loc[2020].iloc[0])
            mm_sle_ax_fl.set_ylim((1 - np.array([mn, mx])) * sle_2020)
            mm_sle_ax_fl.set_ylabel('$\Delta{}$M (mm SLE) – flowline model',
                                    color=colors[1])
            mm_sle_ax_fl.spines["right"].set_position(("axes", +1.2))
            mm_sle_ax_fl.spines["right"].set_color(colors[1])
            mm_sle_ax_fl.tick_params(axis='y', direction='inout',
                                     colors=colors[1])

    # save to file
    if save_fig:
        # plt.tight_layout()
        fig_path = os.path.join(plots_parent_dir, 'cmip_single_ssp',
                                'frac_vol', 'global',
                                f'cmip_global.pdf')
        fig.savefig(fig_path)


def plot_global_mass_loss_abs_all(ref_year=2020,
                                  save_fig=True, secondary_axis=True):
    """Plot global projections of both models for the same SSP(s)

    One plot for each SSP, containing the projections (average +/- one standard
    deviation of all CMIP models) of the VAS model and the flowline model.
    Thereby, the shared (left) axis shows the fraction of the reference volume
    (volume at the reference year) in percentage. The two right axes show the
    absolute values of ice volume (in mm SLE) separately for each model, again
    in relation to the reference year.
    """
    # create figure with given figure size
    plt.style.use('./igs.mplstyle')
    fig = plt.figure(figsize=(6.929, 6))

    # specify SSPs
    ssps = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
    n_tot = len(ssps)
    n_columns = 2

    # iterate over all given shared socioeconomic pathways
    for i, ssp in enumerate(ssps):
        log.info(f'Creating plots for SSP{ssp[-3:]}')

        # define colors
        colors = np.array(["#e63946", "#457b9d"])

        # create axes
        mm_sle_ax = fig.add_subplot(int(np.ceil(n_tot / n_columns)),
                                    n_columns,
                                    i + 1)

        ylabel = not (i % n_columns)
        xlabel = i >= (n_tot - n_columns)

        text = list()
        text_tpl = '{} model: {:.0f}±{:.0f} mm SLE ({}±{}%)'

        ref_slr = dict()

        # plot projections for both models
        for c, model in zip(colors, ['vas', 'fl']):
            # read relative ice volume records
            frac_vol = pd.read_csv(frac_csv_glob.format(model), index_col=0)
            # subset for given SSP
            frac_vol = frac_vol.loc[:,
                       [ssp in column for column in frac_vol.columns]]
            # compute average and standard deviation
            avg_frac = frac_vol.mean(axis=1).loc[2000:2100]
            std_frac = frac_vol.std(axis=1).loc[2000:2100]

            # read absolute volume
            abs_vol = pd.read_csv(abs_csv_glob.format(model), index_col=0)
            # subset for given SSP
            abs_vol = abs_vol.loc[:,
                      [ssp in column for column in abs_vol.columns]]
            # compute average and standard deviation
            sle_2020 = compute_slr(abs_vol.loc[2020].iloc[0])
            ref_slr[model] = sle_2020
            avg_abs = sle_2020 - compute_slr(
                abs_vol.mean(axis=1).loc[2000:2100])
            std_abs = compute_slr(abs_vol.std(axis=1).loc[2000:2100])

            # plot one std deviation range
            mm_sle_ax.fill_between(avg_abs.index, avg_abs - std_abs,
                                   avg_abs + std_abs, alpha=0.3,
                                   color=c, ls='None')

            # define and store text
            model_name = 'Scaling' if model == 'vas' else 'Flowline'
            text.append(text_tpl.format(model_name,
                                        avg_abs.loc[2100], std_abs.loc[2100],
                                        int((1 - avg_frac.loc[2100]) * 100),
                                        int(std_frac.loc[2100] * 100)))

            # plot the ensemble average
            avg_abs.plot(color=c, lw=2, ax=mm_sle_ax)

        # add labels, legend, etc.
        if not xlabel:
            mm_sle_ax.get_xaxis().set_ticklabels([])
        mm_sle_ax.set_xlabel('')
        if ylabel:
            mm_sle_ax.set_ylabel('$\Delta{}$M (mm SLE)')
        else:
            mm_sle_ax.get_yaxis().set_ticklabels([])
            mm_sle_ax.set_ylabel('')

        mm_sle_ax.text(2005, 150, text[0], color=colors[0], ha='left')
        mm_sle_ax.text(2005, 165, text[1], color=colors[1], ha='left')

        plt.xlim([2000, 2100])
        plt.ylim([-15, 180])
        plt.axhline(0, c='grey', ls=':', lw=0.8)
        # plt.grid()

        # add SSP as text
        mm_sle_ax.text(1, 1.01, f'SSP{ssp[3]}-{float(ssp[4:])/10:.1f}',
                       transform=mm_sle_ax.transAxes, ha='right')

        if secondary_axis and not ylabel:
            # get limits of current y-axis showing the volume fraction
            mn, mx = mm_sle_ax.get_ylim()

            # add secondary y-axis with absolute values for VAS model
            frac_ax_vas = mm_sle_ax.twinx()
            frac_ax_vas.set_ylim(1 - np.array([mn, mx]) / ref_slr['vas'])
            frac_ax_vas.set_ylabel(f'M (rel. to {ref_year})', color=colors[0])
            frac_ax_vas.spines["right"].set_color(colors[0])
            frac_ax_vas.tick_params(axis='y', direction='inout',
                                    colors=colors[0])
            # add secondary y-axis with absolute values for flowline model
            frac_ax_fl = mm_sle_ax.twinx()
            frac_ax_fl.set_ylim(1 - np.array([mn, mx]) / ref_slr['fl'])
            frac_ax_fl.set_ylabel(f'M (rel. to {ref_year})', color=colors[1])
            frac_ax_fl.spines["right"].set_position(("axes", +1.2))
            frac_ax_fl.spines["right"].set_color(colors[1])
            frac_ax_fl.tick_params(axis='y', direction='inout',
                                   colors=colors[1])

    # save to file
    if save_fig:
        # plt.tight_layout()
        fig_path = os.path.join(plots_parent_dir, 'cmip_single_ssp',
                                'abs_vol', 'global',
                                f'cmip_global.pdf')
        fig.savefig(fig_path)


def plot_regional_mass_loss_rel_all(ref_year=2020, n_columns=3,
                                    ssps=('ssp126', 'ssp245', 'ssp370',
                                          'ssp585'),
                                    save_fig=True, secondary_axis=False):
    """Plot regional projections of both models for the same SSP(s) with
    relative mass loss as main (left) axis, in analogy to Figure 6 in
    [Marzeion 2020]_

    Each combination of RGI region and SSP results in one plot, containing the
    projections (average +/- one standard deviation of all CMIP models) of the
    VAS model and the flowline model. Thereby, the shared (left) axis shows the
    fraction of the reference volume (volume at the reference year) in
    percentage. The two right axes show the absolute values of ice volume (in
    mm SLE) separately for each model, again in relation to the reference year.
    """
    plt.style.use('./igs.mplstyle')

    # read rgi region file (sorted with descending 2020 ice volume)
    rgi_reg_df = pd.read_csv(os.path.join(data_dir, 'rgi_reg_df.csv'),
                             index_col=0)
    # get total number of regions
    n_tot = rgi_reg_df.shape[0]

    # iterate over all given shared socioeconomic pathways
    ssps = np.atleast_1d(ssps)
    for ssp in ssps:
        log.info(f'Creating plots for SSP{ssp[-3:]}')

        # create figure
        # mm = 1 / 25.4  # conversion from inch to milli meters
        # fig = plt.figure(figsize=[176 * mm, 200 * mm])
        fig = plt.figure()

        # define colors
        colors = np.array(["#e63946", "#457b9d"])

        # iterate over all rgi regions
        for i, (rgi_reg, row) in enumerate(rgi_reg_df.iterrows()):
            log.info(f'Creating plots for RGI region {rgi_reg}')
            # convert region number from integer to zero leading string
            rgi_reg = f'{rgi_reg:02d}'

            if i < 2:
                n_subplot = i + 1
            else:
                n_subplot = i + 3

            # create axes for subplot
            frac_ax = fig.add_subplot(int(np.ceil(n_tot / n_columns)),
                                      n_columns,
                                      n_subplot)

            # plot y tick and axis labels only on the first plot of each row
            ylabel = not ((n_subplot - 1) % n_columns)
            # plot x ticks and axis label only on the last plot of each column
            xlabel = i >= (n_tot - n_columns)
            # add legend only to first plot
            legend = i == 0

            # plot projections for both models
            for c, model in zip(colors, ['vas', 'fl']):
                # read relative ice volume records
                frac = pd.read_csv(frac_csv_tpl.format(model, rgi_reg, ssp),
                                   index_col=0)
                # compute average and standard deviation
                avg = frac.mean(axis=1).loc[2000:2100]
                std = frac.std(axis=1).loc[2000:2100]

                # plot one std deviation range
                frac_ax.fill_between(avg.index, avg - std, avg + std,
                                     alpha=0.3,
                                     color=c)
                # define label
                model_label = 'VAS' if model == 'vas' else 'Flowline'
                label = model_label + '  ({}%)'.format(
                    int((avg.loc[2100] - 1) * 100))
                # plot the ensemble average
                avg.plot(label=model_label, color=c, lw=1, ax=frac_ax)

            # add labels, legend, etc.
            if not xlabel:
                frac_ax.get_xaxis().set_ticklabels([])
            frac_ax.set_xlabel('')
            if ylabel:
                frac_ax.set_ylabel(f'M (rel. to 2020)')
            else:
                frac_ax.get_yaxis().set_ticklabels([])
                frac_ax.set_ylabel('')
            if legend and False:
                plt.legend(loc='lower left', frameon=False)
            plt.ylim([0, 1.4])
            plt.xlim(2000, 2100)
            plt.axhline(1, c='grey', ls=':', lw=0.8)
            # plt.grid()

            # add region as text
            frac_ax.text(1, 1.012, row.region_name,
                         transform=frac_ax.transAxes, ha='right')

            # add 2020 volume as text
            abs_vas = pd.read_csv(abs_csv_tpl.format('vas', rgi_reg, ssp),
                                  index_col=0)
            sle_vas_2020 = compute_slr(abs_vas.loc[ref_year].iloc[0])
            abs_fl = pd.read_csv(abs_csv_tpl.format('fl', rgi_reg, ssp),
                                 index_col=0)
            sle_fl_2020 = compute_slr(abs_fl.loc[ref_year].iloc[0])

            # if n_subplot > 12:
            #     vas_text_pos = (0.95, 0.95)
            #     ha = 'right'
            #     va = 'top'
            # else:
            #     vas_text_pos = (0.05, 0.15)
            #     ha = 'left'
            #     va = 'bottom'
            #
            # fl_text_pos = (0.05, 0.05)

            n_decimals = int(
                max(0, -np.floor(np.log10(min(sle_vas_2020, sle_fl_2020)))))
            # frac_ax.text(*vas_text_pos, f'Scaling: {sle_vas_2020:.{n_decimals}f} mm SLE',
            #              transform=frac_ax.transAxes, ha=ha, va=va)
            # frac_ax.text(*fl_text_pos, f'Flowline: {sle_fl_2020:.{n_decimals}f} mm SLE',
            #              transform=frac_ax.transAxes, ha='left', va='bottom')

            # frac_ax.text(0.05, 0.25,
            #              f'M 2020',
            #              transform=frac_ax.transAxes)
            frac_ax.text(0.05, 0.15,
                         f'{sle_vas_2020:.{n_decimals}f}',
                         transform=frac_ax.transAxes, color=colors[0])
            frac_ax.text(0.05, 0.05,
                         f'{sle_fl_2020:.{n_decimals}f}',
                         transform=frac_ax.transAxes, color=colors[1])

            if secondary_axis:
                # get limits of current y-axis showing the volume fraction
                mn, mx = frac_ax.get_ylim()

                # secondary y-axis with absolute values (mm SLE) for VAS model
                mm_sle_ax_vas = frac_ax.twinx()
                abs_vol = pd.read_csv(abs_csv_tpl.format('vas', rgi_reg, ssp),
                                      index_col=0)
                sle_2020 = compute_slr(abs_vol.loc[ref_year].iloc[0])
                mm_sle_ax_vas.set_ylim((1 - np.array([mn, mx])) * sle_2020)
                # limit number of ticks/labels to 5 (and thereby number of decimal
                # points to 0.00)
                mm_sle_ax_vas.yaxis.set_major_locator(mticker.MaxNLocator(5))
                mm_sle_ax_vas.set_ylabel(
                    'Sea level rise for the VAS model (mm)')

                # secondary y-axis with absolute values (mm SLE) for flowline model
                mm_sle_ax_fl = frac_ax.twinx()
                abs_vol = pd.read_csv(abs_csv_tpl.format('fl', rgi_reg, ssp),
                                      index_col=0)
                sle_2020 = compute_slr(abs_vol.loc[ref_year].iloc[0])
                mm_sle_ax_fl.set_ylim((1 - np.array([mn, mx])) * sle_2020)
                # limit number of ticks/labels to 5 (and thereby number of decimal
                # points to 0.00)
                mm_sle_ax_fl.yaxis.set_major_locator(mticker.MaxNLocator(5))
                mm_sle_ax_fl.set_ylabel(
                    'Sea level rise for the flowline model (mm)')
                mm_sle_ax_fl.spines["right"].set_position(("axes", +1.2))

        # create custom legend
        ax = fig.add_axes([0.5, 0.8, 0.5, 0.2])
        ax.axis('off')

        # add labels
        label_x = 0.1
        ensemble_y = 0.7
        std_y = ensemble_y - 0.15
        ax.text(label_x, ensemble_y, 'Ensemble mean', ha='left', va='center')
        ax.text(label_x, std_y, '± standard deviation', ha='left', va='center')
        scaling_x = 0.5
        flowline_x = scaling_x + 0.2
        model_y = ensemble_y + 0.15
        ax.text(scaling_x, model_y, 'Scaling', ha='left', va='center')
        ax.text(flowline_x, model_y, 'Flowline', ha='left', va='center')

        t = 'The numbers in the lower left corner represent the\n' \
            'aggregate regional glacier mass in 2020 (mm SLE):\n' \
            'red for the scaling model; blue for the flowline model.'
        ax.text(label_x, 0.4, t, va='top')
        # t = ['The numbers in the lower left corner represent the\n'
        #      'aggregate regional glacier mass in 2020 (mm SLE):\n',
        #      'red for the scaling model; ', 'blue for the flowline model.']
        # rainbow_text(label_x, 0.4, t, ['k', 'r', 'b'], va='top')

        # add a lines representing ensemble mean
        l_lw = 1.5
        l_len = 0.15
        l_xseq = np.array([0, l_len])
        l_yseq = np.repeat(ensemble_y, 2)
        line_scaling = Line2D(scaling_x + l_xseq, l_yseq, lw=l_lw,
                              color=colors[0])
        ax.add_line(line_scaling)
        line_flowline = Line2D(flowline_x + l_xseq, l_yseq, lw=l_lw,
                               color=colors[1])
        ax.add_line(line_flowline)
        # add shading representing +/- sigma band
        r_len = l_len
        r_width = 0.12
        rect_scaling = Rectangle((scaling_x, std_y - r_width / 2), r_len,
                                 r_width, alpha=0.3, color=colors[0])
        ax.add_patch(rect_scaling)
        rect_flowline = Rectangle((flowline_x, std_y - r_width / 2), r_len,
                                  r_width, alpha=0.3, color=colors[1])
        ax.add_patch(rect_flowline)

        # save to file
        if save_fig:
            # plt.tight_layout()
            fig_path = os.path.join(plots_parent_dir, 'cmip_single_ssp',
                                    'frac_vol', 'regional',
                                    f'cmip_{ssp}.pdf')
            fig.savefig(fig_path)


def plot_regional_mass_loss_abs_all(ref_year=2020, n_columns=3,
                                    ssps=('ssp126', 'ssp245', 'ssp370',
                                          'ssp585'),
                                    save_fig=True, secondary_axis=False):
    """Plot regional projections of both models for the same SSP(s) with
    relative mass loss as main (left) axis, in analogy to Figure 6 in
    [Marzeion 2020]_

    Each combination of RGI region and SSP results in one plot, containing the
    projections (average +/- one standard deviation of all CMIP models) of the
    VAS model and the flowline model. Thereby, the shared (left) axis shows the
    fraction of the reference volume (volume at the reference year) in
    percentage. The two right axes show the absolute values of ice volume (in
    mm SLE) separately for each model, again in relation to the reference year.
    """
    plt.style.use('./igs.mplstyle')

    # read rgi region file (sorted with descending 2020 ice volume)
    rgi_reg_df = pd.read_csv(os.path.join(data_dir, 'rgi_reg_df.csv'),
                             index_col=0)
    # get total number of regions
    n_tot = rgi_reg_df.shape[0]

    # read y limits from file
    limits = pd.read_csv(os.path.join(data_dir, 'regional_ylimits.csv'), index_col=0)

    # iterate over all given shared socioeconomic pathways
    ssps = np.atleast_1d(ssps)
    for ssp in ssps:
        log.info(f'Creating plots for SSP{ssp[-3:]}')

        # create figure
        # mm = 1 / 25.4  # conversion from inch to milli meters
        # fig = plt.figure(figsize=[176 * mm, 200 * mm])
        fig = plt.figure()

        # define colors
        colors = np.array(["#e63946", "#457b9d"])

        # limits = dict()

        # iterate over all rgi regions
        for i, (rgi_reg, row) in enumerate(rgi_reg_df.iterrows()):
            log.info(f'Creating plots for RGI region {rgi_reg}')
            # convert region number from integer to zero leading string
            rgi_reg = f'{rgi_reg:02d}'

            if i < 2:
                n_subplot = i + 1
            else:
                n_subplot = i + 3

            # create axes for subplot
            mm_sle_ax = fig.add_subplot(int(np.ceil(n_tot / n_columns)),
                                        n_columns,
                                        n_subplot)

            # plot y tick and axis labels only on the first plot of each row
            ylabel = not ((n_subplot - 1) % n_columns)
            # plot x ticks and axis label only on the last plot of each column
            xlabel = i >= (n_tot - n_columns)
            # add legend only to first plot
            legend = i == 0

            # create empty container to store reference values
            avg_end = dict()
            std_end = dict()
            text = dict()

            # plot projections for both models
            for c, model in zip(colors, ['vas', 'fl']):
                # read relative ice volume records
                frac_vol = pd.read_csv(frac_csv_tpl.format(model, rgi_reg, ssp),
                                   index_col=0)
                # compute average and standard deviation
                avg_frac = frac_vol.mean(axis=1).loc[2000:2100]
                std_frac = frac_vol.std(axis=1).loc[2000:2100]

                # read absolute volume
                abs_vol = pd.read_csv(abs_csv_tpl.format(model, rgi_reg, ssp), index_col=0)
                # get reference volume (mm SLE) and store for later use
                ref_vol = abs_vol.loc[ref_year]
                # compute sea level rise contribution relative to ref. year
                mm_slr = compute_slr(ref_vol - abs_vol)
                # compute average and standard deviation
                avg_abs = mm_slr.mean(axis=1).loc[2000:2100]
                std_abs = mm_slr.std(axis=1).loc[2000:2100]

                # plot one std deviation range
                mm_sle_ax.fill_between(avg_abs.index, avg_abs - std_abs,
                                       avg_abs + std_abs, alpha=0.3,
                                       color=c, ls='None')
                # plot the ensemble average
                avg_abs.plot(color=c, lw=2, ax=mm_sle_ax)

                # fill text template and add to container
                avg_end[model] = avg_abs.loc[2100]
                std_end[model] = std_abs.loc[2100]

            # add labels, legend, etc.
            if not xlabel:
                mm_sle_ax.get_xaxis().set_ticklabels([])
            mm_sle_ax.set_xlabel('')
            if ylabel:
                mm_sle_ax.set_ylabel('$\Delta{}$M (mm SLE)')
            else:
                # mm_sle_ax.get_yaxis().set_ticklabels([])
                mm_sle_ax.set_ylabel('')
            if legend and False:
                plt.legend(loc='lower left', frameon=False)
            plt.ylim(limits.loc[int(rgi_reg)])
            plt.xlim(2000, 2100)
            # plt.axhline(0, c='grey', ls=':', lw=0.8)
            # plt.grid()

            # add region as text
            mm_sle_ax.text(1, 1.012, row.region_name,
                           transform=mm_sle_ax.transAxes, ha='right')

            # add 2020 volume as text
            # abs_vas = pd.read_csv(abs_csv_tpl.format('vas', rgi_reg, ssp),
            #                       index_col=0)
            # sle_vas_2020 = compute_slr(abs_vas.loc[ref_year].iloc[0])
            # abs_fl = pd.read_csv(abs_csv_tpl.format('fl', rgi_reg, ssp),
            #                      index_col=0)
            # sle_fl_2020 = compute_slr(abs_fl.loc[ref_year].iloc[0])

            # if n_subplot > 12:
            #     vas_text_pos = (0.95, 0.95)
            #     ha = 'right'
            #     va = 'top'
            # else:
            #     vas_text_pos = (0.05, 0.15)
            #     ha = 'left'
            #     va = 'bottom'
            #
            # fl_text_pos = (0.05, 0.05)

            # n_decimals = int(
            #     max(0, -np.floor(np.log10(min(sle_vas_2020, sle_fl_2020)))))
            # mm_sle_ax.text(*vas_text_pos, f'Scaling: {sle_vas_2020:.{n_decimals}f} mm SLE',
            #              transform=mm_sle_ax.transAxes, ha=ha, va=va)
            # mm_sle_ax.text(*fl_text_pos, f'Flowline: {sle_fl_2020:.{n_decimals}f} mm SLE',
            #              transform=mm_sle_ax.transAxes, ha='left', va='bottom')

            # mm_sle_ax.text(0.05, 0.25,
            #              f'M 2020',
            #              transform=mm_sle_ax.transAxes)
            # mm_sle_ax.text(0.05, 0.15,
            #                f'{sle_vas_2020:.{n_decimals}f}',
            #                transform=mm_sle_ax.transAxes, color=colors[0])
            # mm_sle_ax.text(0.05, 0.05,
            #                f'{sle_fl_2020:.{n_decimals}f}',
            #                transform=mm_sle_ax.transAxes, color=colors[1])

            if n_subplot <= 16:
                t_x = 0.1
                fl_y = 0.9
                vas_y = 0.8
                ha = 'left'
            else:
                t_x = 0.9
                fl_y = 0.2
                vas_y = 0.1
                ha = 'right'

            n_dec_avg = int(max(0, -np.floor(np.log10(avg_end[min(avg_end)]))))
            n_dec_std = int(max(0, -np.floor(np.log10(std_end[min(std_end)]))))
            t = f"{avg_end['vas']:.{n_dec_avg}f}±{std_end['vas']:.{n_dec_std}f}"
            mm_sle_ax.text(t_x, vas_y, t, transform=mm_sle_ax.transAxes, c=colors[0], ha=ha, va='center')
            t = f"{avg_end['fl']:.{n_dec_avg}f}±{std_end['fl']:.{n_dec_std}f}"
            mm_sle_ax.text(t_x, fl_y, t, transform=mm_sle_ax.transAxes, c=colors[1], ha=ha, va='center')

            if secondary_axis and False:
                # get limits of current y-axis showing the volume fraction
                mn, mx = mm_sle_ax.get_ylim()

                # secondary y-axis with absolute values (mm SLE) for VAS model
                mm_sle_ax_vas = mm_sle_ax.twinx()
                abs_vol = pd.read_csv(abs_csv_tpl.format('vas', rgi_reg, ssp),
                                      index_col=0)
                sle_2020 = compute_slr(abs_vol.loc[ref_year].iloc[0])
                mm_sle_ax_vas.set_ylim((1 - np.array([mn, mx])) * sle_2020)
                # limit number of ticks/labels to 5 (and thereby number of decimal
                # points to 0.00)
                mm_sle_ax_vas.yaxis.set_major_locator(mticker.MaxNLocator(5))
                mm_sle_ax_vas.set_ylabel(
                    'Sea level rise for the VAS model (mm)')

                # secondary y-axis with absolute values (mm SLE) for flowline model
                mm_sle_ax_fl = mm_sle_ax.twinx()
                abs_vol = pd.read_csv(abs_csv_tpl.format('fl', rgi_reg, ssp),
                                      index_col=0)
                sle_2020 = compute_slr(abs_vol.loc[ref_year].iloc[0])
                mm_sle_ax_fl.set_ylim((1 - np.array([mn, mx])) * sle_2020)
                # limit number of ticks/labels to 5 (and thereby number of decimal
                # points to 0.00)
                mm_sle_ax_fl.yaxis.set_major_locator(mticker.MaxNLocator(5))
                mm_sle_ax_fl.set_ylabel(
                    'Sea level rise for the flowline model (mm)')
                mm_sle_ax_fl.spines["right"].set_position(("axes", +1.2))

            # store y-axis limits
            # if ssp == 'ssp585':
            #     limits[rgi_reg] = mm_sle_ax.get_ylim()

        # store limits to file
        # if limits:
        #     pd.DataFrame(limits).T.to_csv(os.path.join(data_dir, 'regional_ylimits.csv'))

        # create custom legend
        ax = fig.add_axes([0.5, 0.8, 0.5, 0.2])
        ax.axis('off')

        # add labels
        label_x = 0.1
        ensemble_y = 0.7
        std_y = ensemble_y - 0.15
        ax.text(label_x, ensemble_y, 'Ensemble mean', ha='left', va='center')
        ax.text(label_x, std_y, '± standard deviation', ha='left', va='center')
        scaling_x = 0.5
        flowline_x = scaling_x + 0.2
        model_y = ensemble_y + 0.15
        ax.text(scaling_x, model_y, 'Scaling', ha='left', va='center')
        ax.text(flowline_x, model_y, 'Flowline', ha='left', va='center')

        t = 'The final ensemble mean and standard deviation\n' \
            'in 2100 are annotated as $\overline{x}\pm\sigma$ (mm SLE): red for \n' \
            'the scaling model; blue for the flowline model.'
        ax.text(label_x, 0.4, t, va='top')
        # t = ['The numbers in the lower left corner represent the\n'
        #      'aggregate regional glacier mass in 2020 (mm SLE):\n',
        #      'red for the scaling model; ', 'blue for the flowline model.']
        # rainbow_text(label_x, 0.4, t, ['k', 'r', 'b'], va='top')

        # add a lines representing ensemble mean
        l_lw = 1.5
        l_len = 0.15
        l_xseq = np.array([0, l_len])
        l_yseq = np.repeat(ensemble_y, 2)
        line_scaling = Line2D(scaling_x + l_xseq, l_yseq, lw=l_lw,
                              color=colors[0])
        ax.add_line(line_scaling)
        line_flowline = Line2D(flowline_x + l_xseq, l_yseq, lw=l_lw,
                               color=colors[1])
        ax.add_line(line_flowline)
        # add shading representing +/- sigma band
        r_len = l_len
        r_width = 0.12
        rect_scaling = Rectangle((scaling_x, std_y - r_width / 2), r_len,
                                 r_width, alpha=0.3, color=colors[0])
        ax.add_patch(rect_scaling)
        rect_flowline = Rectangle((flowline_x, std_y - r_width / 2), r_len,
                                  r_width, alpha=0.3, color=colors[1])
        ax.add_patch(rect_flowline)

        # save to file
        if save_fig:
            # plt.tight_layout()
            fig_path = os.path.join(plots_parent_dir, 'cmip_single_ssp',
                                    'abs_vol', 'regional',
                                    f'cmip_{ssp}.pdf')
            fig.savefig(fig_path)
            log.info(f'Storing figure to {fig_path}')


if __name__ == '__main__':

    from matplotlib import font_manager

    font_dirs = ['/home/users/moberrauch/fonts/optima']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    plot_regional_mass_loss_rel_all(n_columns=4)
    plot_regional_mass_loss_abs_all(n_columns=4)
    plot_global_mass_loss_abs_all()
    plot_global_mass_loss_rel_all()
    # plot_regional_mass_loss_abs()
    # plot_global_mass_loss_abs()
