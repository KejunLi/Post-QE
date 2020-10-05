#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
from fit_functions import quadratic_fct
import matplotlib.colors as mc
import colorsys
import scipy.constants as spc



def var_color(color, brightness_offset):
    """
    lightens the given color by multiplying (1-luminosity)
    by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.
    Examples:
    var_color("g", 0.3)
    var_color("#F034A3", 0.6)
    var_color((0.3,0.55,0.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    # print(colorsys.hls_to_rgb(c[0], 1-brightness_offset*(1-c[1]), c[2]))
    return(colorsys.hls_to_rgb(c[0], 1-brightness_offset*(1-c[1]), c[2]))
    

def cfe(*args):
    """
    ============================================================================
    +   Charge formation energy (CFE) and charge transition level (CTL)
    +   E_f = Etot_xq - Etot_bulk + \sum(n\mu) + q(vbm+\mu_e) + E_corr
    +   arg = [Etot_xq, Etot_bulk, q, vbm, cbm, E_corr]
    +   E_f: formation energy of defect x with charge q
    +   Etot_xq: total energy of defect x with charge q
    +   Etot_bulk: total energy of pristine bulk cell
    +   \sum(n\mu): chemical potentials of additional atoms (set to 0)
    +   q: number of charge carriers
    +   \mu_e: electron chemical potential, 0 < \mu < cbm-vbm
    +   E_corr: correction term
    ============================================================================
    """
    mu_e = np.zeros((len(args), 1000), dtype="float")
    E_f = np.zeros((len(args), 1000), dtype="float")
    Etot_xq_corr = np.zeros(len(args), dtype="float")
    for i, arg in enumerate(args):
        mu_e[i, :] = np.linspace(0, arg[4]-arg[3], 1000)
        E_f[i, :] = arg[0] - arg[1] + arg[2] * ( arg[3] + mu_e[i, :] ) + arg[5]
        Etot_xq_corr[i] = arg[0] + arg[5]
    return(mu_e, E_f, Etot_xq_corr)


def plot_ctl(*args):
    """
    ============================================================================
    +   Default format of input:
    +   plot_sg_diagram(*args)
    +   args = [arg[0], arg[1], ...]
    +   arg[0]: E_tot (q = -2, -1, 0, +1, +2)
    +   arg[1]: E_corr (q = -2, -1, 0, +1, +2)
    +   arg[2]: vac (electrostatic potential of defect system)
    +   arg[3]: vbm of pristine hBN referenced to vacuum
    +   arg[4]: cbm of pristine hBN referenced to vacuum
    +   arg[5]: plot range in y-axis
    +   arg[6]: labels
    +   arg[7]: title
    ============================================================================
    """
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sections = len(args)
    bar_width = 1.0 / sections / 2 
    ylim_top = np.amax(list(np.array(args[:])[:, 5]))  # top ylim of plot
    ylim_bot = np.amin(list(np.array(args[:])[:, 5]))  # bottom ylim of plot
    
    num_lines = sections - 1    # number of lines that partition plot
    coord_labels = np.zeros(sections, dtype=float)
    for i in range(num_lines):
        ax.axvline(
            1.0/sections*(i+1), color="k", 
            linestyle="solid", linewidth=0.8
            )
    for i in range(sections):
        coord_labels[i] = bar_width * (i * 2 + 1)
    
    for i, arg in enumerate(args):
        x = [(i+0.1)*bar_width*2, (i+0.9)*bar_width*2]
        E_tot_corr = np.asarray(arg[0]) + np.asarray(arg[1])
        y = E_tot_corr[:-1] - E_tot_corr[1:] - arg[2]
        for j in range(len(y)):
            ax.plot(x, [y[j], y[j]], linewidth=1, color='black')
            ax.text(
                (i+0.1)*bar_width*2,
                y[j],
                arg[6][j],
                horizontalalignment='center', 
                verticalalignment='bottom'
                )            
        ax.fill_between(
            np.array([i,i+1])*bar_width*2,  # x range to fill
            [arg[4], arg[4]],               # lower limit of y; cbm
            [ylim_top, ylim_top],           # upper limit of y
            facecolor="tab:red",            # The fill color
            alpha=0.32,                     # Transparency of the fill
            )
        ax.fill_between(
            np.array([i,i+1])*bar_width*2,  # x range to fill
            [ylim_bot, ylim_bot],           # lower limit of y
            [arg[3], arg[3]],               # upper limit of y; vbm
            facecolor="tab:blue",           # The fill color
            alpha=0.32                      # Transparency of the fill
            )
    ax.set_xlim([0, 1])
    ax.set_ylim([ylim_bot, ylim_top])
    ax.set_ylabel("CTL (eV)")
    plt.xticks(coord_labels, list(np.array(args[:])[:, 7]))
    plt.subplots_adjust(
        left=0.16, bottom=0.08, right=0.98, top=0.98,
        wspace=None, hspace=None
        )
        # end and save figure
    fig.savefig("ctl.pdf")




def plot_sg_diag(*args, mode=1, show_values=True, **kwargs):
    """
    ============================================================================
    +   Default format of input:
    +   plot_sg_diagram(*args)
    +   args = [arg1, arg2, ...]
    +   arg[i] = {
            E_spinu, occ_spinu, E_spind, occ_spind, vac, 
            show_values, vbm, cbm
            }
    +   arg[0]: spin up levels' number
    +   arg[1]: E_spinu
    +   arg[2]: occ_spinu 
    +   arg[3]: spin down levels' number
    +   arg[4]: E_spind
    +   arg[5]: occ_spind
    +   arg[6]: vac (electrostatic potential of defect system)
    +   arg[7]: vbm of pristine hBN referenced to vacuum
    +   arg[8]: cbm of pristine hBN referenced to vacuum
    +   arg[9]: plot range in y-axis
    +   arg[10]: label
    +
    +   kwargs: additional_label = [texts, coords]
    +   texts = ["1a'\u2191", "1a''\u2193", ...]
    +   coords = [(0.01, -5.5), (0.27, -4.5), ...]
    +
    +   mode 1: combine all single-particle diagrams together (default)
    +   mode 2: separately plot each single-particle diagram
    +   show_values: show levels, occupations and eigenvalues (defaul True)
    ============================================================================
    """
    if mode == 1:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        sections = len(args)
        bar_width = 1.0 / sections / 2 
        
        ylim_top = np.amax(list(np.array(args[:])[:, 9]))  # top ylim of plot
        ylim_bot = np.amin(list(np.array(args[:])[:, 9]))  # bottom ylim of plot
        
        num_lines = sections - 1    # number of lines that partition plot
        coord_labels = np.zeros(sections, dtype=float)
        for i in range(num_lines):
            ax.axvline(
                1.0/sections*(i+1), color="k", 
                linestyle="solid", linewidth=0.8
                )
        for i in range(sections):
            coord_labels[i] = bar_width * (i * 2 + 1)

        for i, arg in enumerate(args):
            for j, yval in enumerate(arg[1]):   # spin up
                x = [(i+0.18)*bar_width*2, (i+0.38)*bar_width*2]
                y = [yval-arg[6], yval-arg[6]]
                ax.plot(x, y, color='black')
                # show levels' number, occupations and eigenvalues
                if show_values:
                    if arg[2][j] == 1:  # occupied spin up states
                        ax.text(
                            (i+0.1)*bar_width*2,
                            yval-arg[6],    # referenced to vacuum
                            str(arg[0][j])+"\u2b06|"+
                            str((format(yval-arg[6], ".4f"))),
                            fontsize=8,
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                    else:   # unoccupied spin up states
                        ax.text(
                            (i+0.1)*bar_width*2,
                            yval-arg[6],    # referenced to vacuum
                            str(arg[0][j])+"\u21e7|"+
                            str((format(yval-arg[6], ".4f"))),
                            fontsize=8,
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                # not show levels' number, occupations and eigenvalues
                else:
                    if arg[2][j] == 1:  # occupied spin down states
                        ax.text(
                            (i+0.25+j%2*0.05)*bar_width*2,
                            yval-arg[6],    # referenced to vacuum
                            "\u2b06",
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                    else:   # unoccupied spin down states
                        ax.text(
                            (i+0.25+j%2*0.05)*bar_width*2,
                            yval-arg[6],    # referenced to vacuum
                            "\u21e7",
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
            for j, yval in enumerate(arg[4]):   # spin down
                x = [(i+0.62)*bar_width*2, (i+0.82)*bar_width*2]
                y = [yval-arg[6], yval-arg[6]]
                ax.plot(x, y, color='black')
                # show levels' number, occupations and eigenvalues
                if show_values:
                    if arg[5][j] == 1:  # occupied spin down states
                        ax.text(
                            (i+0.9)*bar_width*2,
                            yval-arg[6],    # referenced to vacuum
                            str(arg[3][j])+"\u2b07|"+
                            str((format(yval-arg[6], ".4f"))),
                            fontsize=8,
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                    else:   # unoccupied spin down states
                        ax.text(
                            (i+0.9)*bar_width*2,
                            yval-arg[6],    # referenced to vacuum
                            str(arg[3][j])+"\u21e9|"+
                            str((format(yval-arg[6], ".4f"))),
                            fontsize=8,
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                # not show levels' number, occupations and eigenvalues
                else:
                    if arg[5][j] == 1:  # occupied spin down states
                        ax.text(
                            (i+0.7+j%2*0.05)*bar_width*2,
                            yval-arg[6],    # referenced to vacuum
                            "\u2b07",
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                    else:   # unoccupied spin down states
                        ax.text(
                            (i+0.7+j%2*0.05)*bar_width*2,
                            yval-arg[6],    # referenced to vacuum
                            "\u21e9",
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
            
            ax.fill_between(
                np.array([i,i+1])*bar_width*2,  # x range to fill
                [arg[8], arg[8]],               # lower limit of y; cbm
                [ylim_top, ylim_top],                       # upper limit of y
                facecolor="tab:red",             # The fill color
                alpha=0.32,                      # Transparency of the fill
                )
            ax.fill_between(
                np.array([i,i+1])*bar_width*2,  # x range to fill
                [ylim_bot, ylim_bot],                     # lower limit of y
                [arg[7], arg[7]],               # upper limit of y; vbm
                facecolor="tab:blue",               # The fill color
                alpha=0.32                      # Transparency of the fill
                )
        if "additional_label" in kwargs:
            texts = kwargs.get("additional_label")[0]
            coords = kwargs.get("additional_label")[1]
            for i in range(len(texts)):
                plt.text(coords[i][0], coords[i][1], texts[i])

        ax.set_xlim([0, 1])
        ax.set_ylim([ylim_bot, ylim_top])
        plt.xticks(coord_labels, list(np.array(args[:])[:, 10]))
        plt.subplots_adjust(
            left=0.08, bottom=0.08, right=0.98, top=0.98,
            wspace=None, hspace=None
            )
    if mode == 2:
        fig, ax = plt.subplots(nrows=1, ncols=len(args))#, constrained_layout=True)
        for i, arg in enumerate(args):
            for j, yval in enumerate(arg[1]):
                x = [0.1, 0.45]
                y = [yval-arg[6], yval-arg[6]]
                ax[i].plot(x, y, color='black')
                # show levels' number, occupations and eigenvalues
                if show_values:
                    if arg[2][j] == 1:  # occupied spin up states
                        ax[i].text(
                            0.1,
                            yval-arg[6],    # referenced to vacuum
                            str(arg[0][j])+"\u2b06|"+
                            str((format(yval-arg[6], ".4f"))),
                            fontsize=8,
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                    else:   # unoccupied spin up states
                        ax[i].text(
                            0.1,
                            yval-arg[6],
                            str(arg[0][j])+"\u21e7|"+
                            str((format(yval-arg[6], ".4f"))),
                            fontsize=8,
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                # not show levels' number, occupations and eigenvalues
                else:
                    if arg[2][j] == 1:  # occupied spin down states
                        ax[i].text(
                            0.25+j%2*0.05,
                            yval-arg[6],    # referenced to vacuum
                            "\u2b06",
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                    else:   # unoccupied spin down states
                        ax[i].text(
                            0.25+j%2*0.05,
                            yval-arg[6],    # referenced to vacuum
                            "\u21e7",
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
            for j, yval in enumerate(arg[4]):   # spin down
                x = [0.55, 0.9]
                y = [yval-arg[6], yval-arg[6]]
                ax[i].plot(x, y, color='black')
                # show levels' number, occupations and eigenvalues
                if show_values:
                    if arg[5][j] == 1:  # occupied spin down states
                        ax[i].text(
                            0.9,
                            yval-arg[6],    # referenced to vacuum
                            str(arg[3][j])+"\u2b07|"+
                            str((format(yval-arg[6], ".4f"))),
                            fontsize=8,
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                    else:   # unoccupied spin down states
                        ax[i].text(
                            0.9,
                            yval-arg[6],    # referenced to vacuum
                            str(arg[3][j])+"\u21e9|"+
                            str((format(yval-arg[6], ".4f"))),
                            fontsize=8,
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                # not show levels' number, occupations and eigenvalues
                else:
                    if arg[5][j] == 1:  # occupied spin down states
                        ax[i].text(
                            0.7+j%2*0.05,
                            yval-arg[6],    # referenced to vacuum
                            "\u2b07",
                            family='cursive',   # font 'cursive' for arrows
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
                    else:
                        ax[i].text(
                            0.7+j%2*0.05,
                            yval-arg[6],    # referenced to vacuum
                            "\u21e9",
                            # font 'DejaVu Sans' for arrows
                            family='DejaVu Sans',
                            horizontalalignment='center', 
                            verticalalignment='center'
                            )
            ax[i].set_xlim([0, 1])
            ax[i].set_ylim(arg[9])
            ax[i].fill_between(
                np.array([0, 1]),  # x range to fill
                [arg[8], arg[8]],               # lower limit of y; cbm
                [np.amax(arg[9]), np.amax(arg[9])],      # upper limit of y
                facecolor="tab:red",             # The fill color
                alpha=0.32,                      # Transparency of the fill
                )
            ax[i].fill_between(
                np.array([0, 1]),  # x range to fill
                [np.amin(arg[9]), np.amin(arg[9])],    # lower limit of y
                [arg[7], arg[7]],               # upper limit of y; vbm
                facecolor="tab:blue",               # The fill color
                alpha=0.32                      # Transparency of the fill
                )
            ax[i].xaxis.set_major_locator(plt.NullLocator())
            ax[0].set_ylabel("E (eV)")
            ax[i].set_title(arg[10])
            
            plt.subplots_adjust(
                left=0.08, bottom=0.02, right=0.98, top=0.9,
                wspace=None, hspace=None
                )
    # end and save figure
    fig.savefig("sg-diag.pdf")



def plot_config_coord_diag(etot_1, etot_2, dQ_1, dQ_2, xlim, ylim, **kwargs):
    """
    ============================================================================
    +   etot_1: the arrays of total energies of state 1
    +   etot_2: the arrays of total energies of state 2
    +   dQ_1: change of nuclear coordinate of state 1
    +   dQ_2: change of nuclear coordinate of state 2
    +   xlim: limitation of x axis for plot
    +   ylim: limitation of y axis for plot
    +   kwargs: arrows, labels
    +   arrows == {"left_arrow_shift": val, "right_arrow_shift": val,
                "elongate": val, "E_zpl_shift": value, "E_rel_shift": val}
    +   labels == {"label1": {"x": xval, "y": yval, "name": "??"},
                "label2": {"x": xval, "y": yval, "name": "??"}}
    ============================================================================
    """
    ev2J = spc.physical_constants["Hartree energy in eV"][0]/2
    kB = spc.Boltzmann
    T_room = 300 # K

    npoints = len(etot_1)
    min_etot = min(min(etot_1), min(etot_2))
    # set etot_1 and dQ_1 as the energy surface and change of 
    # nuclear coordinates of the state with lower energy
    if min(etot_1) - min(etot_2) < 0:
        pass
    else:
        temp_etot = etot_1
        etot_1 = etot_2
        etot_2 = temp_etot
        temp_dQ = dQ_1
        dQ_1 = dQ_2
        dQ_2 = temp_dQ
    sec_min_etot = min(etot_2)
    max_etot = max(etot_2)
    sec_max_etot = max(etot_1)
    for i in range(npoints):
        if etot_1[i] == min_etot:
            x_of_min_etot = dQ_1[i]
        if etot_2[i] == sec_min_etot:
            x_of_sec_min_etot = dQ_2[i]

    E_zpl = float(format(sec_min_etot - min_etot, ".5f"))
    E_rel = float(format(sec_max_etot - min_etot, ".5f"))
    E_abs = float(format(max_etot - min_etot, ".5f"))
    E_em = float(format(sec_min_etot - sec_max_etot, ".5f"))
    print("\rData from calculation:")
    print("\rE_zpl = {} eV".format(E_zpl)) # ZPL
    print("\rE_rel = {} eV".format(E_rel)) # The energy of gs in es geometry
    print("\rE_abs = {} eV".format(E_abs)) # absorption
    print("\rE_em = {} eV\n\n".format(E_em)) # emission

    # this part does plotting
    fit_vals_1 = np.polyfit(dQ_1[0:5], etot_1[0:5], 2)
    fit_vals_2 = np.polyfit(dQ_2[7:14], etot_2[7:14], 2)
    min_etot_fit = quadratic_fct(
        x_of_min_etot, fit_vals_1[0], fit_vals_1[1], fit_vals_1[2]
        )
    sec_max_etot_fit = quadratic_fct(
        x_of_sec_min_etot, fit_vals_1[0], fit_vals_1[1], fit_vals_1[2]
        )
    sec_min_etot_fit = quadratic_fct(
        x_of_sec_min_etot, fit_vals_2[0], fit_vals_2[1], fit_vals_2[2]
        )
    max_etot_fit = quadratic_fct(
        x_of_min_etot, fit_vals_2[0], fit_vals_2[1], fit_vals_2[2]
        )
    E_rel_fit = float(format(sec_max_etot_fit - min_etot_fit, ".5f"))
    E_zpl_fit = float(format(sec_min_etot_fit - min_etot_fit, ".5f"))

    plt.axhline(0.0, linestyle="dashed", color="black")
    plt.axhline(E_zpl_fit, linestyle="dashed", color="black")

    plt.axvline(x_of_min_etot, linestyle="dashed", color="black")
    plt.axvline(x_of_sec_min_etot, linestyle="dashed", color="black")
    
    min_x = min(xlim)
    max_x = max(xlim)
    x = np.arange(min_x, max_x, 0.001)
    y_1 = np.zeros(len(x), dtype=float)
    y_2 = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        y_1[i] = quadratic_fct(
            x[i], fit_vals_1[0], fit_vals_1[1], fit_vals_1[2]
            ) - min_etot
    for i in range(len(x)):
        y_2[i] = quadratic_fct(
            x[i], fit_vals_2[0], fit_vals_2[1], fit_vals_2[2]
            ) - min_etot

    if "arrows" in kwargs:
        values = kwargs.get("arrows")
        plt.annotate(
            '', # text
            xy=(
                x_of_min_etot-values["left_arrow_shift"],\
                0-values["elongate"]
                ), # the point (x, y) to annotate
            xytext=(
                x_of_min_etot-values["left_arrow_shift"],\
                E_zpl_fit+values["elongate"]
                ), # the position to place the text at. If None, defaults to (x, y)
            arrowprops=dict(arrowstyle="<|-|>", color = "k")
            )
        plt.hlines(
            E_rel_fit, x_of_sec_min_etot,
            x_of_sec_min_etot+values["right_arrow_shift"],
            linestyles="dashed"
            )
        plt.annotate(
            '',
            xy=(
                x_of_sec_min_etot+values["right_arrow_shift"],\
                0-values["elongate"]
                ),
            xytext=(
                x_of_sec_min_etot+values["right_arrow_shift"],\
                E_rel_fit+values["elongate"]
                ),
            arrowprops=dict(arrowstyle="<|-|>", color = "k")
            )
        plt.text(
            x_of_min_etot-values["E_zpl_shift"],
            E_zpl_fit/2.0, "\u0394E"
            )
        plt.text(
            x_of_sec_min_etot+values["E_rel_shift"],
            E_rel_fit/2.0, "\u0394E$_{rel}$"
            )

    if "labels" in kwargs:
        labels = kwargs.get("labels")
        plt.text(
            labels["label1"]["x"], labels["label1"]["y"],
            labels["label1"]["name"]
            )
        plt.text(
            labels["label2"]["x"], labels["label2"]["y"],
            labels["label2"]["name"]
            )

    plt.plot(x, y_1, linewidth=2, color="tab:blue")
    plt.plot(x, y_2, linewidth=2, color="tab:red")

    for i in range(npoints):
        plt.plot(
            dQ_1[i], etot_1[i]-min_etot, marker="o", markersize=6,
            markerfacecolor="w", color="tab:blue"
            )
        plt.plot(
            dQ_2[i], etot_2[i]-min_etot, marker="o", markersize=6,
            markerfacecolor="w", color="tab:red"
            )

    E_abs_fit = float(format(max_etot_fit - min_etot_fit, ".5f"))
    E_em_fit = float(format(sec_min_etot_fit - sec_max_etot_fit, ".5f"))
    print("\rData from fitting:")
    print("\rE_zpl_fit = {} eV".format(E_zpl_fit)) # ZPL
    print("\rE_rel_fit = {} eV".format(E_rel_fit)) # The energy of gs in es geometry
    print("\rE_abs_fit = {} eV".format(E_abs_fit)) # absorption
    print("\rE_em_fit = {} eV \n".format(E_em_fit)) # emission

    y_diff = min(np.abs(np.array(y_2)-np.array(y_1)))
    for i in range(len(x)):
        if np.abs(y_2[i] - y_1[i]) == y_diff:
            E_barrier = float(format(y_2[i] - min(y_2), ".6f"))
    # estimated transition rate through energy barrier
    rate_eff = np.power(10.0,12) * np.exp(-E_barrier*ev2J/(kB*T_room))
    time_eff = 1.0/rate_eff
    rate_eff = "{:.5e}".format(rate_eff)
    time_eff = "{:.5e}".format(time_eff)
    print("\rE_barrier = {} eV".format(E_barrier))
    print("\rrate_eff = {} s^-1".format(rate_eff))
    print("\rtime_eff = {} s\n".format(time_eff))
    


def view_3d(atomic_positions, atomic_mass, **kwargs):
    fig = plt.figure(
        num=None, figsize=(10, 7.5), dpi=120, facecolor='w', edgecolor='k'
    )
    ax = fig.add_subplot(1,1,1, projection="3d")
    ax.scatter(
        atomic_positions[:, 0], atomic_positions[:, 1], atomic_positions[:, 2], 
        zdir="z", s=atomic_mass*2, c=atomic_mass/10
    )
    surf = ax.plot_trisurf(
        atomic_positions[:, 0], atomic_positions[:, 1], atomic_positions[:, 2], 
        linewidth=0.2, antialiased=True, cmap=plt.cm.viridis, alpha=0.1
    )

    if "circle" in kwargs:
        alpha = np.linspace(0.0, np.pi*2, 200)
        (center, radius) = kwargs.get("circle")
        x1 = radius * np.cos(alpha) + center[0]
        y1 = radius * np.sin(alpha) + center[1]
        z1 = center[2]
        ax.scatter(center[0], center[1], center[2])
        ax.plot(x1, y1, z1)
    else:
        sys.stdout.write(
            "\rTip: to show circle with defects as center, " +
            "specify 'circle = (r, x0, y0, z0)' from input"
            )

    sys.stdout.write("\nPlot \rTips")
    # title
    if "title" in kwargs:
        if kwargs.get("title") == None:
            sys.stdout.write("\rNo title")
        else:
            ax.set_title("{}".format(kwargs.get("title")))
    else:
        sys.stdout.write(
            "\rTip: to add title to diagram, specify " +
            "'title = {}'".format("?") + "from input"
            )

    # about view directions and axes labels
    if "view_direction" in kwargs:
        if kwargs.get("view_direction") == "front_view":
            ax.view_init(azim=-90, elev=0)
            ax.w_yaxis.line.set_lw(0.)
            ax.set_yticks([])
            if "xlabel" in kwargs:
                if kwargs.get("xlabel") == None:
                    sys.stdout.write("\rNo xlabel")
                else:
                    ax.set_xlabel(kwargs.get("xlabel"))
            else:
                sys.stdout.write(
                    "\rTip: to show xlabel, specify 'xlabel = ?' from input"
                    )
            if "zlabel" in kwargs:
                if kwargs.get("zlabel") == None:
                    sys.stdout.write("\rNo zlabel")
                else:
                    ax.set_zlabel(kwargs.get("zlabel"))
            else:
                sys.stdout.write(
                    "\rTip: to show zlabel, specify 'zlabel = ?' from input"
                    )
        elif kwargs.get("view_direction") == "top_view":
            ax.view_init(azim=-90, elev=90)
            ax.w_zaxis.line.set_lw(0.)
            ax.set_zticks([])
            if "xlabel" in kwargs:
                if kwargs.get("xlabel") == None:
                    sys.stdout.write("\rNo xlabel")
                else:
                    ax.set_xlabel(kwargs.get("xlabel"))
            else:
                sys.stdout.write(
                    "\rTip: to show xlabel, specify 'xlabel = ?' from input"
                    )
            if "ylabel" in kwargs:
                if kwargs.get("ylabel") == None:
                    sys.stdout.write("\rNo ylabel")
                else:
                    ax.set_ylabel(kwargs.get("ylabel"))
            else:
                sys.stdout.write(
                    "\rTip: to show ylabel, specify 'ylabel = ?' from input"
                    )
        else:
            ax.view_init(azim=-180, elev=0)
            ax.w_xaxis.line.set_lw(0.)
            ax.set_xticks([])
            if "ylabel" in kwargs:
                if kwargs.get("ylabel") == None:
                    sys.stdout.write("\rNo ylabel")
                else:
                    ax.set_ylabel(kwargs.get("ylabel"))
            else:
                sys.stdout.write(
                    "\rTip: to show ylabel, specify 'ylabel = ?' from input"
                    )
            if "zlabel" in kwargs:
                if kwargs.get("zlabel") == None:
                    sys.stdout.write("\rNo zlabel")
                else:
                    ax.set_zlabel(kwargs.get("zlabel"))
            else:
                sys.stdout.write(
                    "\rTip: to show zlabel, specify 'zlabel = ?' from input"
                    )
    else:
        sys.stdout.write(
            "\rTip: to take the top view of the 3D plot, " +
            "specify 'view_direction = top_view' from input, so as " +
            "for front view and left view."
            )
        if "xlabel" in kwargs:
            if kwargs.get("xlabel") == None:
                sys.stdout.write("\rNo xlabel")
            else:
                ax.set_xlabel(kwargs.get("xlabel"))
        else:
            sys.stdout.write(
                "\rTip: to show xlabel, specify " +
                "'xlabel = ?' from input"
                )
        if "ylabel" in kwargs:
            if kwargs.get("ylabel") == None:
                sys.stdout.write("\rNo ylabel")
            else:
                ax.set_ylabel(kwargs.get("ylabel"))
        else:
            sys.stdout.write(
                "\rTip: to show ylabel, specify 'ylabel = ?' from input"
                )
        if "zlabel" in kwargs:
            if kwargs.get("zlabel") == None:
                sys.stdout.write("\rNo zlabel")
            else:
                ax.set_zlabel(kwargs.get("zlabel"))
        else:
            sys.stdout.write(
                "\rTip: to show zlabel, specify 'zlabel = ?' from input"
                )

    # range of axes
    if "xlim" in kwargs:
        if kwargs.get("xlim") == None:
            sys.stdout.write("\rNo xlim")
        else:
            ax.set_xlim(kwargs.get("xlim"))
    else:
        sys.stdout.write(
            "\rTip: to constrain x-axis, specify 'xlim = (min,max)' from input"
            )
    if "ylim" in kwargs:
        if kwargs.get("ylim") == None:
            sys.stdout.write("\rNo ylim")
        else:
            ax.set_ylim(kwargs.get("ylim"))
    else:
        sys.stdout.write(
            "\rTip: to constrain y-axis, specify 'ylim = (min,max)' from input"
            )
    if "zlim" in kwargs:
        if kwargs.get("zlim") == None:
            sys.stdout.write("\rNo zlim")
        else:
            ax.set_zlim(kwargs.get("zlim"))
    else:
        sys.stdout.write(
            "\rTip: to constrain z-axis, specify 'zlim = (min,max)' from input"
            )

    # colorbar
    if "plot_corlorbar" in kwargs:
        if kwargs.get("plot_corlorbar"):
            if "setup_boundaries" in kwargs:
                fig.colorbar(surf, boundaries=kwargs.get("setup_boundaries"))
            else:
                sys.stdout.write(
                    "\rTip: to set boundaries for colorbar, " +
                    "specify 'setup_boundaries = np.linspace(min,max)' " +
                    "from input"
                    )
                fig.colorbar(surf)
    else:
        sys.stdout.write(
            "\rTip: to plot colorbar, specify " +
            "'plot_corlorbar = True' from input"
            )
    sys.stdout.flush()

