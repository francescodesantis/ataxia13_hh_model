# ==========================================================
# SIMULATION CONSTANTS
# ==========================================================
SIM_DURATION = 50
DT           = 0.001

REC_VARS = [
    "V_m",
    # Kv3.1 gating
    "n", "p",
    # K-LT gating
    "l", "r",
    # Na gating
    "m", "h",
]

# ==========================================================
# PARAMETER GROUPS
# All defaults taken directly from hh_macica.nestml (corrected version).
# { group_label: [(param, default, lo, hi, resolution, tooltip), ...] }
# ==========================================================
PARAM_GROUPS = {
    # ----------------------------------------------------------
    "Membrane": [
        ("V_m_init", -64.5,  -90.0, -40.0, 0.1,
         "Initial membrane potential (mV)"),
        ("C_m",       22.36,   1.0,  200.0, 0.01,
         "Membrane capacitance (pF)"),
        ("E_L",       -63.0,  -80.0, -40.0, 0.1,
         "Leak reversal potential (mV)"),
        ("g_L",         2.0,    0.0,  50.0, 0.01,
         "Leak conductance (nS)"),
    ],

    # ----------------------------------------------------------
    "Na+ — Conductance & Reversal": [
        ("g_Na",   500.0,    0.0, 5000.0, 1.0,
         "Na+ max conductance (nS)"),
        ("E_Na",    50.0,   20.0,   80.0, 0.1,
         "Na+ reversal potential (mV)"),
    ],

    # ----------------------------------------------------------
    "Na+ m-gate  alpha_m / beta_m": [
        ("k_alpha_m",   76.4,   0.1,  500.0, 0.1,
         "alpha_m rate prefactor  [k*exp(eta*V)]"),
        ("eta_alpha_m",  0.037, -0.2,    0.2, 0.001,
         "alpha_m exponential slope (mV-1) — positive => faster with depol."),
        ("k_beta_m",     6.93,  0.1,  200.0, 0.01,
         "beta_m rate prefactor"),
        ("eta_beta_m",  -0.043, -0.2,   0.2, 0.001,
         "beta_m exponential slope (mV-1) — negative => faster with hyperpol."),
    ],

    # ----------------------------------------------------------
    "Na+ h-gate  alpha_h / beta_h": [
        ("k_alpha_h",   0.000533, 0.0,    0.05,  0.000001,
         "alpha_h rate prefactor"),
        ("eta_alpha_h", -0.0909, -0.3,    0.0,   0.001,
         "alpha_h exponential slope (mV-1) — negative => inactivation at depol."),
        ("k_beta_h",     7.87,    0.1,   50.0,   0.01,
         "beta_h rate prefactor"),
        ("eta_beta_h",   0.0691,  0.0,    0.3,   0.001,
         "beta_h exponential slope (mV-1)"),
    ],

    # ----------------------------------------------------------
    "Kv3.1 — Conductance & Reversal": [
        ("g_Kv3_1",  150.0,   0.0, 2000.0, 1.0,
         "Kv3.1 max conductance (nS)  <- primary KCNC3 / SCA13 target"),
        ("E_Kv3_1",  -80.0, -100.0,  -40.0, 0.1,
         "Kv3.1 reversal potential (mV)"),
    ],

    # ----------------------------------------------------------
    "Kv3.1 n-gate  alpha_n / beta_n": [
        ("k_alpha_n",   0.2719,  0.001,  2.0,   0.0001,
         "alpha_n rate prefactor"),
        ("eta_alpha_n", 0.04,   -0.1,    0.2,   0.001,
         "alpha_n exponential slope (mV-1)"),
        ("k_beta_n",    0.1974,  0.001,  2.0,   0.0001,
         "beta_n rate prefactor"),
        ("eta_beta_n",  0.0,    -0.1,    0.1,   0.001,
         "beta_n exponential slope (mV-1) — 0 => voltage-independent deactivation"),
    ],

    # ----------------------------------------------------------
    "Kv3.1 p-gate  alpha_p / beta_p  (slow modulation)": [
        ("k_alpha_p",   0.00713, 0.0001,  0.1,   0.00001,
         "alpha_p prefactor — small => slow; p spans 80-100% due to 0.2 weight in I_Kv3_1"),
        ("eta_alpha_p", -0.1942, -0.5,    0.0,   0.001,
         "alpha_p slope (mV-1) — negative => gate closes with depolarisation"),
        ("k_beta_p",    0.0935,  0.001,   1.0,   0.0001,
         "beta_p rate prefactor"),
        ("eta_beta_p",  0.0058,  -0.05,   0.05,  0.0001,
         "beta_p slope (mV-1) — near-zero => weak voltage dependence"),
    ],

    # ----------------------------------------------------------
    "K-LT — Conductance & Reversal": [
        ("g_K_LT",  20.0,   0.0, 500.0, 0.5,
         "K-LT max conductance (nS)"),
        ("E_K_LT",  -80.0, -100.0, -40.0, 0.1,
         "K-LT reversal potential (mV)"),
    ],

    # ----------------------------------------------------------
    "K-LT l-gate  alpha_l / beta_l  (fast activation)": [
        ("k_alpha_l",   1.2,    0.01,  10.0,  0.01,
         "alpha_l rate prefactor — large => fast activation"),
        ("eta_alpha_l", 0.03512, 0.0,   0.15,  0.0001,
         "alpha_l exponential slope (mV-1)"),
        ("k_beta_l",    0.2248,  0.001,  5.0,  0.0001,
         "beta_l rate prefactor"),
        ("eta_beta_l", -0.0319, -0.15,   0.0,  0.0001,
         "beta_l slope (mV-1) — negative => slower deactivation at hyperpol."),
    ],

    # ----------------------------------------------------------
    "K-LT r-gate  alpha_r / beta_r  (slow inactivation)": [
        ("k_alpha_r",   0.0438, 0.001,   1.0,  0.0001,
         "alpha_r rate prefactor"),
        ("eta_alpha_r", -0.0053, -0.1,   0.1,  0.0001,
         "alpha_r slope (mV-1) — weak voltage dependence"),
        ("k_beta_r",    0.0562,  0.001,  1.0,  0.0001,
         "beta_r rate prefactor"),
        ("eta_beta_r", -0.0047, -0.1,    0.1,  0.0001,
         "beta_r slope (mV-1) — weak voltage dependence"),
    ],
}

# ==========================================================
# PRESETS — MNTB Wild-Type only
# All values from the corrected hh_macica.nestml parameters block.
# ==========================================================
PRESETS = {
    "MNTB Wild-Type": {
        # Membrane
        "V_m_init":    -63.0,
        "C_m":          10.0,
        "E_L":          -63,#-66.008,
        "g_L":            2.0,
        "refr_T":         1.0,
        # Na
        "g_Na":         500.0,
        "E_Na":          50.0,
        "k_alpha_m":     76.4,
        "eta_alpha_m":    0.037,
        "k_beta_m":       6.93,
        "eta_beta_m":    -0.043,
        "k_alpha_h":      0.000533,
        "eta_alpha_h":   -0.0909,
        "k_beta_h":       7.87,
        "eta_beta_h":     0.0691,
        # Kv3.1
        "g_Kv3_1":      150.0,
        "E_Kv3_1":      -80.0,
        "k_alpha_n":      0.2719,
        "eta_alpha_n":    0.04,
        "k_beta_n":       0.1974,
        "eta_beta_n":     0.0,
        "k_alpha_p":      0.00713,
        "eta_alpha_p":   -0.1942,
        "k_beta_p":       0.0935,
        "eta_beta_p":     0.0058,
        # K-LT
        "g_K_LT":        20.0,
        "E_K_LT":       -80.0,
        "k_alpha_l":      1.2,
        "eta_alpha_l":    0.03512,
        "k_beta_l":       0.2248,
        "eta_beta_l":    -0.0319,
        "k_alpha_r":      0.0438,
        "eta_alpha_r":   -0.0053,
        "k_beta_r":       0.0562,
        "eta_beta_r":    -0.0047,
        # External
        "I_e": 0.0,  # pA (200 nA)
    },
}

# ==========================================================
# COLORS
# ==========================================================
CP = {
    "Vm":    "#111111",
    "Na":    "#d62728",   # red
    "h":     "#9467bd",   # purple
    "m":     "#ff7f0e",   # orange
    "Kv3_1": "#1f77b4",  # blue
    "K_LT":  "#2ca02c",  # green
    "n":     "#1f77b4",  # blue
    "p":     "#aec7e8",  # light blue
    "l":     "#2ca02c",  # green
    "r":     "#98df8a",  # light green
}
