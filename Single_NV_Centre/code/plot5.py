import numpy as np
import matplotlib.pyplot as plt
import simulate_TStrength_NV_ensemble as simnv

# =============================================================================
# 1. PHYSICAL & EXPERIMENTAL PARAMETERS IN THE LOCAL NV FRAME
# =============================================================================

nfreq = 1000
freqi = 2830.0
freqf = 2910.0
MWfreq = np.linspace(freqi, freqf, nfreq)

B0 = 0.0 
B_local_vec = np.array([0.0, 0.0, B0])

E0_strain = 11e6  
thetaE = np.pi / 2.0
phiE = 0.0
E_local_vec = simnv.get_vector_cartesian(E0_strain, thetaE, phiE)

thetaMW = np.pi / 2.0
phiMW = np.pi / 4.0
MW_local_vec = simnv.get_vector_cartesian(1.0, thetaMW, phiMW)

# Base linewidth at very low power (no power broadening applied yet)
base_linewidth = 1.85  # MHz (this is the intrinsic linewidth)

target_powers_dbm = [5.0, 10.0, 15.0, 19.0]
colors = {5.0: 'purple', 10.0: 'navy', 15.0: 'magenta', 19.0: 'teal'}

plt.figure(figsize=(9, 5))

for p_dbm in target_powers_dbm:
    
    # Calculate scaling factor for contrast only (NOT for linewidth)
    P_mw = 10.0 ** (p_dbm / 10.0)
    P_sat = 15.0
    calculated_scaling = (P_mw / (P_sat + P_mw)) * 1.18
    calculated_scaling = min(calculated_scaling, 1.0)

    # Pass base_linewidth to library - library will handle power broadening internally
    T_strength = simnv.ESR_singleNV(
        MWfreq, 
        MW_local_vec, 
        B_local_vec, 
        E_local_vec, 
        base_linewidth,  # Pass base linewidth, NOT calculated_linewidth
        hyperfine_on=False,
        mw_power_dbm=p_dbm
    )
    
    T_min = np.min(T_strength)
    T_max = np.max(T_strength)
    T_norm = (T_strength - T_min) / (T_max - T_min) if (T_max - T_min) > 0 else T_strength
    
    max_dip_depth = 0.046 * calculated_scaling
    odmr_fluorescence = 1.0 - (max_dip_depth * T_norm)
    
    plt.plot(MWfreq, odmr_fluorescence, label=f"{p_dbm} dBm", color=colors[p_dbm], linewidth=2)

plt.title("Relative change is fluorescence vs. MW frequency for different MW powers\n (B=0.0 G)", fontweight='bold')
plt.xlabel("Microwave frequency [MHz]")
plt.ylabel("Normalized Fluorescence [a.u.]")
plt.ylim(0.94, 1.01)
plt.xlim(freqi, freqf)
plt.grid(True, linestyle=':', alpha=0.6)

plt.axvline(2868.25, color='gray', linestyle='--', alpha=0.5)
plt.axvline(2871.75, color='gray', linestyle='--', alpha=0.5)
plt.text(2870.0, 1, "3-4 MHz split", ha='center', fontweight='bold', fontsize=10)

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()