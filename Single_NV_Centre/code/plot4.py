# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import simulate_TStrength_NV_ensemble as simnv

# Clear any existing plots
plt.clf()
plt.close('all')

# Create figure with specific size and position
fig = plt.figure(figsize=(12, 7))
# Center the figure on screen (optional - works with TkAgg backend)
try:
    fig.canvas.manager.window.move(
        int(fig.canvas.manager.window.winfo_screenwidth()/2 - 600),
        int(fig.canvas.manager.window.winfo_screenheight()/2 - 350)
    )
except:
    pass

nfreq = 4000
center_freq = 2870.0  # Zero-field splitting
range_width = 120     # Total width in MHz
freqi = center_freq - range_width/2  # 2810
freqf = center_freq + range_width/2  # 2930
MWfreq = np.linspace(freqi, freqf, nfreq)

B_fields = [0, 5, 10, 15]

# Strain ON
E0_strain = 5.6e6
thetaE = np.pi / 2.0
phiE = 0.0
E_local_vec = simnv.get_vector_cartesian(E0_strain, thetaE, phiE)

thetaMW = np.pi / 2.0
phiMW = np.pi / 4.0
MW_vec_local = simnv.get_vector_cartesian(1.0, thetaMW, phiMW)

p_dbm = 19.0
P_mw = 10.0 ** (p_dbm / 10.0)

calculated_linewidth = 0.5

colors = ['blue', 'green', 'orange', 'red']

for idx, B0 in enumerate(B_fields):
    B_local_vec = np.array([0.0, 0.0, B0])
    
    T_strength = simnv.ESR_singleNV(
        MWfreq, MW_vec_local, B_local_vec, E_local_vec, 
        calculated_linewidth, hyperfine_on=True, mw_power_dbm=p_dbm
    )
    
    T_min = np.min(T_strength)
    T_max = np.max(T_strength)
    T_norm = (T_strength - T_min) / (T_max - T_min) if (T_max - T_min) > 0 else T_strength
    
    max_dip_depth = 0.046
    odmr_fluorescence = 1.0 - (max_dip_depth * T_norm)
    
    vertical_offset = idx * 0.08
    stacked_trace = odmr_fluorescence - vertical_offset
    
    plt.plot(MWfreq, stacked_trace, linewidth=1.5, color=colors[idx])
    plt.text(freqf + 3, 1.0 - vertical_offset, f"{B0} G", va='center', fontweight='bold', fontsize=10, color=colors[idx])

# Set symmetric x-axis limits (no extra padding)
plt.xlim(freqi, freqf)

x_ticks_major = np.arange(2810, 2940, 20)
x_ticks_minor = np.arange(2810, 2940, 5)

plt.xticks(x_ticks_major, rotation=45, fontsize=9)
plt.gca().set_xticks(x_ticks_minor, minor=True)

plt.grid(True, which='major', linestyle='-', alpha=0.6)
plt.grid(True, which='minor', linestyle=':', alpha=0.3)

plt.ylabel("")
plt.gca().set_yticklabels([])
plt.gca().set_yticks([])

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

plt.xlabel("Microwave Frequency [MHz]", fontsize=11, fontweight='bold')
plt.title("Single NV ODMR Spectrum vs. Magnetic Field\n(Strain ON, Hyperfine ON, MW power = 19 dBm)", 
          fontsize=12, fontweight='bold', pad=12)

# Adjust layout to use full canvas
plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)

plt.show()