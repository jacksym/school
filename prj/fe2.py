import numpy as np
import matplotlib.pyplot as plt
import data_proc

plt.style.use('seaborn-paper')


# faintest: 0896, 1223    brightest: 0148, 0151
path = "./ogle/"; event = "blg-0011"

ogle1 = data_proc.OGLE_data(path+event)


fig1, ax1 = plt.subplots()
ogle1.plot(ax1, event)
fig1.savefig("./images/phot_dat.png", dpi=300)
print("saved figure phot_dat.png")

fig2, ax2 = plt.subplots()
ogle1.freq_dist_r(ax2)
ax2.invert_xaxis()
gauss_model1 = ogle1.gauss_model(ogle1.meas, ogle1.freqs, ax2)
ogle1.chi_test(ogle1.meas, gauss_model1, ax2)
fig2.savefig("./images/freq_dist.png", dpi=300)
print("saved figure freq_dist.png")

path = "./ogle/"; event = "blg-0040"
trash_fig, trash_ax = plt.subplots()
ogle1 = data_proc.OGLE_data(path+event)
ogle1.plot(trash_ax, event)

fig3, ax3 = plt.subplots()
ogle1.cumu_dist(ogle1.ress, ax3)
gauss_model_c1 = ogle1.gauss_model_c(ogle1.meas_c, ogle1.cdf, ax3)
ogle1.ks_test(ogle1.meas_c, ogle1.cdf, gauss_model_c1, ax3)
ogle1.and_dar_test(ogle1.meas_c, gauss_model_c1, ax3)
fig3.savefig("./images/cumu_dist.png", dpi=300)
print("saved figure cumu_dist.png")

st_fig, st_ax = plt.subplots()
t = np.linspace(-5, 5, 200)
for i in range(1, 10):
    label = r'$\nu = %s$' % i
    st_ax.plot(t, data_proc.student_form(t, i, 0), label=label)
    st_ax.legend()
st_fig.savefig("./images/t_dist.png", dpi=300)
print("saved figure t_dist.png")

fig5, ax5 = plt.subplots()
ogle1.cumu_dist(ogle1.ress, ax5)
gauss_model_c2 = ogle1.gauss_model_c(ogle1.meas_c, ogle1.cdf, ax5)
student_model = ogle1.student_model_c(ogle1.meas_c, ogle1.cdf, ax5)
ogle1.ks_test(ogle1.meas_c, ogle1.cdf, student_model, ax5)
ogle1.and_dar_test(ogle1.meas_c, student_model, ax5)
ax5.set_title("CDFs of data, normal, and t-distributions")
ax5.legend(loc='center right')
fig5.savefig("./images/t_cdf.png", dpi=300)

lo_fig, lo_ax = plt.subplots()
x = np.linspace(-5, 5, 200)
for i in range(1, 10):
    label = r'$\gamma = %s$' % i
    lo_ax.plot(t, data_proc.cauchy_form(x, 0, i), label=label)
lo_ax.legend()
lo_fig.savefig("./images/lo_dist.png", dpi=300)
print("saved figure lo_dist.png")


fig6, ax6 = plt.subplots()
ogle1.cumu_dist(ogle1.ress, ax6)
ogle1.y_string = 0.95
gauss_model_c2 = ogle1.gauss_model_c(ogle1.meas_c, ogle1.cdf, ax6)
cauchy_model = ogle1.cauchy_model_c(ogle1.meas_c, ogle1.cdf, ax6)
ogle1.ks_test(ogle1.meas_c, ogle1.cdf, cauchy_model, ax6)
ogle1.and_dar_test(ogle1.meas_c, cauchy_model, ax6)
ax6.set_title("CDFs of data, normal, and Cauchy distributions")
ax6.legend(loc='center right')
fig6.savefig("./images/cauchy_cdf.png", dpi=300)
fig6.show()
print("saved figure cauchy_cdf.png")


fig7, ax7 = plt.subplots()
ogle1.cumu_dist(ogle1.ress, ax7)
new_meas = ogle1.man_err(ogle1.diffs, ogle1.err_const, 0.1)
ogle1.cumu_dist(new_meas, ax7)
ax7.arrow(-1.3, 0.15, 0.5, 0, linewidth=3, head_length=0.3, head_width=0.03)
ax7.arrow(1.6, 0.93, -0.5, 0, linewidth=3, head_length=0.3, head_width=0.03)
fig7.savefig("./images/err_cor1.png", dpi=300)
print("saved figure err_cor1.png")
fig7.show()

fig8, ax8 = plt.subplots()
gauss_ec_model = ogle1.gauss_model_ec(ogle1.diffs, ax8)
ogle1.ks_test(ogle1.meas_ec, ogle1.cdf_ec, gauss_ec_model, ax8)
ogle1.and_dar_test(ogle1.meas_ec, gauss_ec_model, ax8)
fig8.savefig("./images/gauss_ec.png", dpi=300)
print("saved figure gauss_ec.png")


final_fig, final_ax = plt.subplots()
df = np.loadtxt("error_cor.dat")
mean_brights = df[:,0]
corrections = df[:,1]
m, b = np.polyfit(mean_brights, corrections, 1)
fit_string = r'$C(m) \sim %.4E\ m  %.4E$' % (m, b)
final_ax.text(19, 0.175, fit_string)
fit = m*mean_brights + b
final_ax.plot(mean_brights, fit, linestyle='--', color='blue')
final_ax.scatter(mean_brights, corrections, color='brown', marker='.')
final_ax.set_ylim(0)
final_ax.set_xlim(np.max(mean_brights), np.min(mean_brights))
final_ax.set_title("Error Corrections for Average Magnitude")
final_fig.savefig("./images/final.png", dpi=300)
