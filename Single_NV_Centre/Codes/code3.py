# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import simulate_ODMR_NV as simnv

# Clear any existing plots
plt.clf()
plt.close('all')

# Create figure with specific size and position
fig = plt.figure(figsize=(14, 8))
# Center the figure on screen
try:
    fig.canvas.manager.window.move(
        int(fig.canvas.manager.window.winfo_screenwidth()/2 - 700),
        int(fig.canvas.manager.window.winfo_screenheight()/2 - 400)
    )
except:
    pass

nfreq = 8000  # Increased resolution for better separation
center_freq = 2870.0
range_width = 100  # Narrower range to focus on hyperfine structure (2820 to 2920)
freqi = center_freq - range_width/2  # 2820
freqf = center_freq + range_width/2  # 2920
MWfreq = np.linspace(freqi, freqf, nfreq)

B_fields = [0, 5, 10, 15]

E_local_vec = np.array([0.0, 0.0, 0.0])

thetaMW = np.pi / 2.0
phiMW = np.pi / 4.0
MW_vec_local = simnv.get_vector_cartesian(1.0, thetaMW, phiMW)

p_dbm = 19.0
P_mw = 10.0 ** (p_dbm / 10.0)

# CRITICAL: Use very narrow linewidth to resolve hyperfine (2-3 MHz splitting)
# Linewidth must be smaller than hyperfine splitting (~2.14 MHz)
calculated_linewidth = 0.3  # MHz (narrow enough to resolve hyperfine)

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
    
    vertical_offset = idx * 0.10  # Slightly larger offset for clarity
    stacked_trace = odmr_fluorescence - vertical_offset
    
    plt.plot(MWfreq, stacked_trace, linewidth=1.5, color=colors[idx])
    plt.text(freqf + 2, 1.0 - vertical_offset, f"{B0} G", va='center', fontweight='bold', fontsize=10, color=colors[idx])

# Set symmetric x-axis limits
plt.xlim(freqi, freqf)

# Finer x-axis ticks for hyperfine structure
x_ticks_major = np.arange(2820, 2930, 10)
x_ticks_minor = np.arange(2820, 2930, 2)

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
plt.title("Single NV ODMR Spectrum vs. Magnetic Field\n(Strain OFF, Hyperfine ON, MW power = 19 dBm)",
          fontsize=12, fontweight='bold', pad=12)

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08)

plt.show()