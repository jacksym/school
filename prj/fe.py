#Microlensing Project
#Jack Symonds 
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import data_proc

# faintest: 0896, 1223    brightest: 0148, 0151
path = "./ogle/"; event = "blg-0011"

ogle1 = data_proc.OGLE_data(path+event)

fig, ax = plt.subplots(figsize=(10, 7))
ogle1.plot(ax, event)
# ogle1.plot_f(ax, event)
# plt.close()


# fig2, (ax21, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 5]})
# ogle1.const_dist(ax21, ax2)
# ogle1.gauss_model(ogle1.bins, ogle1.bars, ax2)
# ogle1.student_model(ogle1.bins, ogle1.bars, ax2)

# fig2, ax2 = plt.subplots()
# ogle1.freq_dist(ax2)
# gauss_model = ogle1.gauss_model(ogle1.meas, ogle1.freqs, ax2)
# ogle1.student_model(ogle1.meas, ogle1.freqs, ax2)
# ogle1.cauchy_model(ogle1.meas, ogle1.freqs, ax2)
# chi = ogle1.chi_test(ogle1.ress, gauss_model, ax2)


# fig3, ax3 = plt.subplots()
# ogle1.cumu_dist(ax3)

# gauss_model = ogle1.gauss_model_c(ogle1.meas_c, ogle1.cdf, ax3)
# ogle1.ks_test(ogle1.meas_c, ogle1.cdf, gauss_model, ax3)
# ogle1.and_dar_test(gauss_model, ax3)

# student_model = ogle1.student_model_c(ogle1.meas_c, ogle1.cdf, ax3)

# cauchy_model = ogle1.cauchy_model_c(ogle1.meas_c, ogle1.cdf, ax3)
# ogle1.ks_test(cauchy_model, ax3)
# ogle1.and_dar_test(cauchy_model, ax3)

# fig4, ax4 = plt.subplots()
# ogle1.qqplot(ax4, ogle1.gauss_model_dist)

# fig3, (ax31, ax3) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
# ogle1.model_deviation(ax31, ax3)
# ogle1.gauss_model(ogle1.bins_c, ogle1.bars_c, ax3)

# fig5, ax5 = plt.subplots()
# gauss_model_ec = ogle1.gauss_model_ec(ogle1.diffs, ax5)
# ogle1.ks_test(gauss_model_ec, ax5)
# ogle1.and_dar_test(gauss_model_ec, ax5)

plt.show()
