import numpy as np
import pandas as pd
import ampworks as amp
import matplotlib.pyplot as plt


# %% Import the data
# ==================
# Calculating LAM and LLI by fitting dQdV curves requires low-rate data from
# negative and positive electrode half cells, as well as for the full cell. The
# data must be a pandas DataFrame (or an ampworks Dataset). All data must have
# state of charge and voltage columns labeled 'SOC' and 'Volts', respectively.
# The full cell data also requires a capacity column labeled 'Ah', which is a
# required input for the final LAM and LLI calculations.

# Below we import the negative and positive electrode data, which has already
# been pre-processed into 'SOC' and 'Volts' columns. Furthermore, the half-cell
# datasets imported here are already smoothed. You may need to perform your own
# smoothing on your raw data, which we also demonstrate below for the full-cell
# data. However, there are many ways to smooth data, and the demonstrated method
# is just one option, and is not guaranteed to be the best choice for your data.

neg = pd.read_csv('dqdv_data/gr.csv')  # negative electrode data
pos = pd.read_csv('dqdv_data/nmc.csv')  # positive electrode data

# The full cell data is also imported below. Here we import raw data from two
# reference performance tests; one at the beginning of life (BOL, cycle 2) and
# the other at the end of life (EOL, cycle 3866). While the half-cell data is
# already smoothed, we will demonstrate how to smooth the full cell data below.

raw2 = pd.read_csv('dqdv_data/charge2.csv')
raw3866 = pd.read_csv('dqdv_data/charge3866.csv')


# %% Plot the data
# ================
# The fitting routine expects the 'SOC' columns for all datasets to be in the
# same reference frame as the full cell. That is, as the full cell SOC changes
# from 0 to 1, the negative electrode delithiates and the positive electrode
# lithiates. Therefore, by convention here, the negative electrode voltage is
# expected to decrease with increasing SOC, whereas the positive electrode and
# full cell voltages are expected to increase with increasing SOC. The fitting
# routine does perform an internal check for this convention, but it is good to
# visualize your data and attempt to enforce this convention yourself as a best
# practice. The plot below shows our data meets this convention.

fig, axs = plt.subplots(2, 2, figsize=[8, 5], sharex=True)

axs[0, 0].plot(neg['SOC'], neg['Volts'])
axs[0, 1].plot(pos['SOC'], pos['Volts'])
axs[1, 0].plot(raw2['SOC'], raw2['Volts'])
axs[1, 1].plot(raw3866['SOC'], raw3866['Volts'])

labels = ['(a) neg', '(b) pos', '(c) cell BOL', '(d) cell EOL']
for ax, label in zip(axs.ravel(), labels):
    amp.plotutils.add_text(ax, 0.1, 0.8, label)

for ax in axs[:, 0]:
    ax.set_ylabel('Voltage [V]')
for ax in axs[1, :]:
    ax.set_xlabel('SOC [-]')

amp.plotutils.format_ticks(axs)

plt.show()


# %% Smooth data with splines
# ===========================
# We smooth the full-cell data at BOL and EOL using the 'DqdvSpline' class. This
# class takes in raw data and fits a B-Spline to the voltage vs. capacity curve.
# The fitted spline can then be evaluated at any SOC. Prior to constructing the
# spline, we convert the pandas DataFrame into an ampworks Dataset so that we
# can downsample the voltage by a specified resolution. This allows us to plot
# the numerical dQdV and dVdQ derivatives against the spline and make sure all
# are in good agreement.
def smooth_data(data, volts_resolution, smoothing_factor, plot=False):
    data = amp.Dataset(data.copy())
    data = data.downsample('Volts', resolution=volts_resolution)
    spline = amp.dqdv.DqdvSpline().fit(data, s=smoothing_factor)

    if plot:
        spline.plot()

    soc = np.linspace(0, 1, 501)
    volts = spline.volts_(soc)
    Ah = np.interp(soc, spline.SOC_, spline.Ah_)
    return pd.DataFrame({'Ah': Ah, 'Volts': volts, 'SOC': soc})


cell_BOL = smooth_data(raw2, 4e-3, 2e-4, plot=True)
cell_EOL = smooth_data(raw3866, 4e-3, 1e-5, plot=True)

plt.show()


# %% Create a fitter instance
# ===========================
# The 'DqdvFitter' class is used to hold and fit the data. Aside from inputing
# electrode and full cell data you can choose how you'd like the optimization
# to run by setting 'cost_terms'. By default, the objective function minimizes
# errors across voltage, dqdv, and dvdq simultaneously. However, you can also
# choose to minimize based on any subset of these three. Note that 'cost_terms'
# can also be changed after initialization, as shown below. The datasets can
# also be replaced by re-setting any of the 'neg', 'pos', or 'cell' properties.

fitter = amp.dqdv.DqdvFitter(neg, pos, cell_BOL)
fitter.cost_terms = ['voltage', 'dqdv', 'dvdq']


# %% Grid searches
# ================
# Because fitting routines can get stuck in local minima, it can be important
# to have a good initial guess. The 'grid_search()' method helps with this.
# Given a number of discretizations, it applies a brute force method to find
# a good initial guess by discretizing the xmin/xmax regions into Nx points,
# and evaluating all physically realistic locations (i.e., xmax > xmin). The
# result isn't always great, but is typically good enough to use as a starting
# value for a more robust fitting routine. You can also see what the plot of
# the best fit looks like using the 'plot()' method, which takes a fit result.

fitres_tmp = fitter.grid_search(21)
fitter.plot(fitres_tmp.x)
print(fitres_tmp, "\n")

plt.show()


# %% Constrained fits
# ===================
# The 'constrained_fit()' method executes a routine from scipy.optimize to find
# values of xmin/xmax (and and iR offset if 'voltage' is in 'cost_terms'). The
# routine forces xmax > xmin for each electrode and sets bounds (+/-) on each
# xmin and xmax based on the 'bounds' keyword argument. See the docstrings for
# more information and detail on changing some of the optimization options.

# The 'constrained_fit()' method takes in a starting guess. You can pass the
# fit result from the 'grid_search()' if you ran one. Otherwise, you can start
# with the 'constrained_fit()' routine right way and pass the output from a
# previous routine back in to see if the fit continues to improve. Or just try
# [0, 1, 0, 1] as an initial guess and see what happens. This assumes xmin/xmax
# are 0 and 1 for both electrodes. The order of inputs is [xn0, xn1, xp0, xp1],
# where 'n' and 'p' refer to the negative and positive electrodes, respectively.

fitres2 = fitter.constrained_fit(fitres_tmp.x)
fitter.plot(fitres2.x)
print(fitres2, "\n")

plt.show()


# %% Swapping to another data set
# ===============================
# There is no need to create a 'fitter' instance for multiple files if you are
# batch processing data. Instead, fit the full cell data starting at beginning
# of life (BOL) and move toward end of life (EOL). A guess from the previous
# fit is typically good enough that there is no need to re-run 'grid_search()'.

fitter.cell = cell_EOL

fitres3866 = fitter.constrained_fit(fitres2.x)
fitter.plot(fitres3866.x)
print(fitres3866, "\n")

plt.show()


# %% Calculating LAM/LLI
# ======================
# If your main purpose for the dQdV fitting is to calculate loss of active
# material (LAM) and loss of lithium inventory (LLI) then you will need to
# loop over and collect the fitted stoichiometries from many cell datasets
# throughout life. Use 'DqdvFitTable' to store all of your results. Then call
# 'calc_lam_lli' and 'plot_lam_lli' to calculate and/or visualize degradation
# modes. Simply initialize an instance of 'DqdvFitTable' before you loop
# over all of your fits, and append the fit result to the table instance after
# each fit is completed. For example, below we make an instance and add the
# fitres2 and fitres3866 results. The fitres_tmp is skipped because it was
# only performed to give a better starting guess for the constrained fit that
# provided fitres2. 'DqdvFitTable' also allows you to track some of your
# own metrics as well via an 'extra_cols' argument. This can be used to have
# columns like 'days', 'efc', 'cycle_number', etc. that you might want to keep
# track of for plotting or fitting life models to later. You can access all of
# the results info via the tables 'df' property, which is just a standard
# pandas DataFrame. This gives you access to save the results, add columns in
# post-processing steps, etc. Below we plot the LAM and LLI values vs. cycle
# number rather than defaulting to the dataframe's row indices, demonstrating
# how to use the 'extra_cols' argument. In the plot, we also show how to include
# approximated standard deviations for the LAM and LLI values. These come from
# propagating the uncertainties from the fitted x0/x1 values. However, be sure
# to still check your fits before you draw too many conclusions from these
# standard deviation estimates, as bad fits stuck in a local minimum can give
# low uncertainties in x0/x1 that should not be trusted. These can then give
# low LAM/LLI uncertainties that also should not be trusted.

fit_table = amp.dqdv.DqdvFitTable(extra_cols=['Cycle'])
fit_table.append(fitres2, Cycle=2)
fit_table.append(fitres3866, Cycle=3866)

deg_table = amp.dqdv.calc_lam_lli(fit_table)
amp.dqdv.plot_lam_lli(deg_table, x_col='Cycle', std=True)

plt.show()

# optionally save fit_table and deg_table
# fit_table.df.to_csv(...)
# deg_table.df.to_csv(...)


# %% Electrode voltages
# =====================
# If you performed the fitting routine to gather information on the electrode
# voltage limits then you can calculate those using the following commands.
# Note that these calculations do not require the full cell capacity info, so
# you can simply use a dumpy value if needed, like cell_BOL['Ah'] = 1.

neg_vlims = fitter.get_ocv('neg', fitres2.x[0:2])
pos_vlims = fitter.get_ocv('pos', fitres2.x[2:4])

print(f"neg voltage limits: {neg_vlims.min():.3f}--{neg_vlims.max():.3f} V")
print(f"pos voltage limits: {pos_vlims.min():.3f}--{pos_vlims.max():.3f} V")

# If you'd prefer to plot the full voltage windows of each electrode, you can
# do that as well, as shown below. Note that when we plot the negative elctrode
# voltage window we use `1.0 - neg_soc` to flip the reference frame back to the
# negative electrode's perspective. This only needs to get done when plotting,
# since the `get_ocv()` method evaluates in the full-cell reference frame so it
# it compatible with the `fitres.x` values without adjustments.

neg_soc = np.linspace(fitres2.x[0], fitres2.x[1], 101)
pos_soc = np.linspace(fitres2.x[2], fitres2.x[3], 101)

fig, axs = plt.subplots(1, 2, figsize=[8, 3])

axs[0].plot(1.0 - neg_soc, fitter.get_ocv('neg', neg_soc))
axs[1].plot(pos_soc, fitter.get_ocv('pos', pos_soc))

for ax in axs:
    ax.set_xlabel('SOC [-]')
    ax.set_ylabel('Voltage [V]')

amp.plotutils.format_ticks(axs)

plt.show()


# %% Using a GUI
# ==============
# If you installed ampworks with the optional GUI dependencies (either by using
# pip install ampworks[gui] or pip install .[dev]), then you can also perform
# this analysis using a local web interface. Simply execute the command below
# in your terminal (or Anaconda Prompt) to launch the GUI. It is relatively
# straight forward to use, however, the user guide is not yet available. This
# will be added in a future release as the software matures. Be aware, however,
# that the GUI does not provide pre-processing and smoothing steps. These will
# need to be performed separately before importing the data. If you'd like to
# try the GUI with the example data given here, make sure to use the smoothed
# full cell data (charge2_smooth.csv and charge3866_smooth.csv) rather than the
# raw data (charge2.csv and charge3866.csv).

# ampworks --app dQdV

# If you are using Jupyter Notebooks you can also launch the GUI from any code
# cell using the following function. There are a couple inputs you can use to
# allow the GUI to run within Jupyter or to launch an external tab. You can
# also control the app height if you choose to run the GUI within a Jupyter
# cell.

# amp.dqdv.run_gui()

# To calculate LAM and LLI degradation modes after using the GUI to fit your
# data, you can import the fit results into a 'DqdvFitTable' instance and then
# use the 'calc_lam_lli' and/or 'plot_lam_lli' methods as shown above. Loading
# the fit results can be done using the 'DqdvFitTable.from_csv()' method. Don't
# forget to add any extra columns you might want to use for plotting, since the
# GUI only stores the bare minimum info needed for LAM/LLI calculations.
