import os
import sys

# ==========================================================
# DYLD_LIBRARY_PATH must be set BEFORE the process starts.
# Re-exec with correct env if needed.
# ==========================================================
NESTML_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hh_macica.nestml")
MODULE_TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hh_macica_module")
MODULE_NAME   = "hh_macica_module"
NEURON_MODEL  = "hh_macica_nestml"

_REEXEC_FLAG = "__HH_MACICA_REEXEC__"

if os.environ.get(_REEXEC_FLAG) != "1":
    dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
    if MODULE_TARGET not in dyld:
        new_env = os.environ.copy()
        new_env["DYLD_LIBRARY_PATH"] = MODULE_TARGET + (":" + dyld if dyld else "")
        new_env[_REEXEC_FLAG] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

# ==========================================================
# IMPORTS
# ==========================================================
import nest
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import find_peaks

from params import (
    SIM_DURATION,
    DT,
    REC_VARS,
    PARAM_GROUPS,
    PRESETS,
    CP,
)

# ==========================================================
# MODEL COMPILATION / INSTALLATION
# ==========================================================
def compile_nestml():
    from pynestml.frontend.pynestml_frontend import generate_nest_target
    import inspect
    sig = inspect.signature(generate_nest_target)
    print(f"[NESTML] generate_nest_target signature: {sig}")
    print("[NESTML] Compiling hh_macica.nestml ...")

    kwargs = dict(
        input_path=NESTML_FILE,
        target_path=MODULE_TARGET,
        module_name=MODULE_NAME,
        logging_level="WARNING",
    )
    accepted = sig.parameters.keys()
    optional = dict(
        suffix="_nestml",
        install_path=MODULE_TARGET,
    )
    for k, v in optional.items():
        if k in accepted:
            kwargs[k] = v

    generate_nest_target(**kwargs)
    print("[NESTML] Compilation done.")

def ensure_module_installed():
    try:
        nest.Install(MODULE_NAME)
        return
    except Exception as e:
        err = str(e).lower()
        if "file not found" not in err and "could not be opened" not in err:
            raise

    compile_nestml()

    try:
        nest.Install(MODULE_NAME)
    except Exception as e:
        print(f"\n[FATAL] Could not install {MODULE_NAME} after compilation:\n  {e}")
        print(f"  DYLD_LIBRARY_PATH should include: {MODULE_TARGET}")
        sys.exit(1)

ensure_module_installed()

# ==========================================================
# AP ANALYSIS
# ==========================================================
def analyze_ap(t, V):
    peaks, _ = find_peaks(V, height=0, prominence=20)
    if len(peaks) == 0:
        return ["No AP detected"]
    results = []
    for pk in peaks:
        v_peak  = V[pk]
        end_idx = min(pk + int(5.0 / DT), len(V) - 1)
        ahp     = np.min(V[pk:end_idx]) - V[0]
        v_half  = (v_peak + V[0]) / 2.0
        try:
            left  = np.where(V[:pk] < v_half)[0][-1]
            right = np.where(V[pk:] < v_half)[0][0] + pk
            hw    = t[right] - t[left]
            results.append(
                f"t={t[pk]:.1f}ms | Peak={v_peak:.1f}mV | HW={hw:.3f}ms | AHP={ahp:.2f}mV"
            )
        except IndexError:
            results.append(
                f"t={t[pk]:.1f}ms | Peak={v_peak:.1f}mV | HW=n/a | AHP={ahp:.2f}mV"
            )
    if len(peaks) > 1:
        isi = np.diff(t[peaks])
        results += [
            f"\nFiring rate : {1000.0 / np.mean(isi):.1f} Hz",
            f"ISI CV      : {np.std(isi)/np.mean(isi):.3f}" if len(isi) > 1 else "ISI CV      : n/a",
            f"Spike count : {len(peaks)}",
        ]
    return results

# ==========================================================
# Na kinetics params — kept as model params but NOT shown as sliders
# ==========================================================
_NA_PARAMS = {
    "k_alpha_m", "eta_alpha_m", "k_beta_m", "eta_beta_m",
    "k_alpha_h", "eta_alpha_h", "k_beta_h", "eta_beta_h",
}

# ==========================================================
# GUI
# ==========================================================
class MacicaGUI:
    def __init__(self, root):
        self.root = root
        root.title("hh_macica Neuron Tuner — Na / Kv3.1 / K-LT")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # ---- Scrollable left panel ----------------------------------
        left_outer = ttk.Frame(root)
        left_outer.grid(row=0, column=0, sticky="ns")

        cv = tk.Canvas(left_outer, width=360, highlightthickness=0)
        sb = ttk.Scrollbar(left_outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)

        ctrl = ttk.Frame(cv, padding=8)
        cv.create_window((0, 0), window=ctrl, anchor="nw")
        ctrl.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind_all("<MouseWheel>",
                    lambda e: cv.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # ---- Preset -------------------------------------------------
        ttk.Label(ctrl, text="hh_macica Neuron Tuner",
                  font=("", 11, "bold")).pack(pady=(0, 4))
        self.preset_var = tk.StringVar(value="MNTB Wild-Type")
        ttk.OptionMenu(ctrl, self.preset_var, "MNTB Wild-Type",
                       *PRESETS.keys(),
                       command=self.load_preset).pack(fill="x", pady=4)

        # ---- Parameter groups (K channels only) ---------------------
        self.vars      = {}
        self._defaults = {}

        # Collect Na defaults silently (needed for nest.Create params)
        for group_name, params in PARAM_GROUPS.items():
            for param, default, *_ in params:
                if param in _NA_PARAMS:
                    self.vars[param]      = tk.DoubleVar(value=default)
                    self._defaults[param] = default

        # Render only non-Na groups
        for group_name, params in PARAM_GROUPS.items():
            visible_params = [(p, d, lo, hi, res, tip)
                              for p, d, lo, hi, res, tip in params
                              if p not in _NA_PARAMS]
            if not visible_params:
                continue

            grp = ttk.LabelFrame(ctrl, text=group_name, padding=5)
            grp.pack(fill="x", pady=3)

            for param, default, lo, hi, res, tooltip in visible_params:
                self.vars[param]      = tk.DoubleVar(value=default)
                self._defaults[param] = default

                row = ttk.Frame(grp)
                row.pack(fill="x", pady=2)

                lbl = ttk.Label(row, text=param, width=14, anchor="w")
                lbl.pack(side="left")
                lbl.bind("<Enter>", lambda e, t=tooltip: self.tooltip.config(text=t))
                lbl.bind("<Leave>", lambda e: self.tooltip.config(text=""))

                ent = ttk.Entry(row, textvariable=self.vars[param], width=9)
                ent.pack(side="right")
                ent.bind("<Return>", lambda e: self.run())

                ttk.Scale(row, from_=lo, to=hi,
                          variable=self.vars[param], orient="horizontal",
                          command=lambda x, p=param, r=res: self._snap(p, r)
                          ).pack(side="left", fill="x", expand=True)

        # ---- Stimulation -------------------------------------------
        stim = ttk.LabelFrame(ctrl, text="Stimulation", padding=5)
        stim.pack(fill="x", pady=4)

        # DC amplitude
        self.dc_var      = tk.DoubleVar(value=200.0)
        self._dc_default = 200.0
        ttk.Label(stim, text="DC Amplitude (pA)").pack(anchor="w")
        amp_row = ttk.Frame(stim)
        amp_row.pack(fill="x")
        ttk.Scale(amp_row, from_=0, to=2000, variable=self.dc_var,
                  orient="horizontal",
                  command=lambda x: self._snap_dc_amp()).pack(side="left", fill="x", expand=True)
        ttk.Entry(amp_row, textvariable=self.dc_var, width=7).pack(side="right")
        self.dc_var.trace_add("write", lambda *_: self.run())

        # DC duration
        self.dc_dur_var      = tk.DoubleVar(value=40.0)   # ms of active injection
        self._dc_dur_default = 40.0
        ttk.Label(stim, text="DC Duration (ms)").pack(anchor="w", pady=(6, 0))
        dur_row = ttk.Frame(stim)
        dur_row.pack(fill="x")
        ttk.Scale(dur_row, from_=1, to=SIM_DURATION - 10, variable=self.dc_dur_var,
                  orient="horizontal",
                  command=lambda x: self._snap_dc_dur()).pack(side="left", fill="x", expand=True)
        ttk.Entry(dur_row, textvariable=self.dc_dur_var, width=7).pack(side="right")
        self.dc_dur_var.trace_add("write", lambda *_: self.run())

        # DC onset (fixed at 5 ms, shown as info label)
        ttk.Label(stim, text="DC onset fixed at 5 ms", foreground="gray",
                  font=("", 8, "italic")).pack(anchor="w")

        # ---- AP analysis --------------------------------------------
        rf = ttk.LabelFrame(ctrl, text="AP Analysis", padding=5)
        rf.pack(fill="both", expand=True, pady=4)
        self.results_txt = tk.Text(rf, height=12, width=36, font=("Courier", 8))
        self.results_txt.pack(fill="both", expand=True)

        # ---- Run / Reset buttons ------------------------------------
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(fill="x", pady=6)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        ttk.Button(btn_frame, text="Run",   command=self.run  ).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(btn_frame, text="Reset", command=self.reset).grid(row=0, column=1, sticky="ew", padx=(2, 0))

        # ---- Tooltip bar -------------------------------------------
        self.tooltip = ttk.Label(root, text="", foreground="gray",
                                 font=("", 8, "italic"))
        self.tooltip.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        # ---- Plots (5 panels, V_m and I_currents enlarged) ----------
        # Row 0: V_m          (tall)
        # Row 1: I_currents   (tall)
        # Row 2: Na gating    (m, h)   — compact
        # Row 3: Kv3.1 gating (n, p)  — compact
        # Row 4: K-LT gating  (l, r)  — compact
        self.fig, self.axes = plt.subplots(
            5, 1, figsize=(7, 13), sharex=True,
            gridspec_kw={"height_ratios": [4, 3, 1.8, 1.8, 1.8]}
        )
        self.fig.tight_layout(pad=2.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        self.run()

    # ---------------------------------------------------------------- #
    def _snap(self, param, res):
        v = round(self.vars[param].get() / res) * res
        self.vars[param].set(v)
        self.run()

    def _snap_dc_amp(self):
        v = round(self.dc_var.get() / 10.0) * 10.0
        self.dc_var.set(v)

    def _snap_dc_dur(self):
        v = round(self.dc_dur_var.get() / 1.0) * 1.0
        self.dc_dur_var.set(v)

    def load_preset(self, name):
        for k, v in PRESETS[name].items():
            if k in self.vars:
                self.vars[k].set(v)
        self.run()

    def reset(self):
        for param, default in self._defaults.items():
            self.vars[param].set(default)
        self.dc_var.set(self._dc_default)
        self.dc_dur_var.set(self._dc_dur_default)
        self.preset_var.set("MNTB Wild-Type")
        self.run()

    # ---------------------------------------------------------------- #
    def run(self):
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": DT, "print_time": False})

        try:
            nest.Install(MODULE_NAME)
        except Exception:
            pass

        params = {k: v.get() for k, v in self.vars.items()}

        try:
            neuron = nest.Create(NEURON_MODEL, params=params)
        except Exception as e:
            self._err(f"nest.Create failed:\n{e}")
            return

        dc_start = 5.0
        dc_stop  = dc_start + max(1.0, self.dc_dur_var.get())

        dc = nest.Create("dc_generator",
                         params={"amplitude": self.dc_var.get(),
                                 "start": dc_start,
                                 "stop":  dc_stop})
        mm = nest.Create("multimeter",
                         params={"record_from": REC_VARS, "interval": DT})
        nest.Connect(dc, neuron)
        nest.Connect(mm, neuron)

        try:
            nest.Simulate(SIM_DURATION)
        except Exception as e:
            self._err(f"Simulation error:\n{e}")
            return

        ev = nest.GetStatus(mm, "events")[0]
        t = ev["times"]
        V = ev["V_m"]
        n, p = ev["n"], ev["p"]
        l, r = ev["l"], ev["r"]
        m, h = ev["m"], ev["h"]

        # ---- Reconstruct ionic currents (pA) -----------------------
        def _g(k):
            return self.vars[k].get()

        I_Na    = _g("g_Na")    * m**3 * h                  * (V - _g("E_Na"))
        I_Kv3_1 = _g("g_Kv3_1")* n**3 * (0.8 + 0.2 * p)   * (_g("E_Kv3_1") - V)
        I_K_LT  = _g("g_K_LT") * l    * r                   * (_g("E_K_LT")  - V)
        I_L     = _g("g_L")                                  * (V - _g("E_L"))

        ax0, ax1, ax2, ax3, ax4 = self.axes

        # ---- V_m (row 0 — tall) ------------------------------------
        ax0.clear()
        ax0.plot(t, V, color=CP["Vm"], lw=1.2)
        ax0.set_ylabel("V_m (mV)", fontsize=9)
        ax0.set_title(
            f"{self.preset_var.get()}   |   DC = {self.dc_var.get():.0f} pA  ×  {self.dc_dur_var.get():.0f} ms",
            fontsize=9
        )
        ax0.axhline(0, color="gray", lw=0.4, ls="--")

        # ---- Ionic currents (row 1 — tall) -------------------------
        ax1.clear()
        ax1.plot(t, I_Na,    label="I_Na    (pA)", color=CP["Na"],    lw=1.0)
        ax1.plot(t, I_Kv3_1, label="I_Kv3.1 (pA)", color=CP["Kv3_1"], lw=1.0)
        ax1.plot(t, I_K_LT,  label="I_K-LT  (pA)", color=CP["K_LT"],  lw=1.0)
        ax1.plot(t, I_L,     label="I_L      (pA)", color=CP["h"],     lw=1.0, ls="--")
        ax1.axhline(0, color="gray", lw=0.4, ls="--")
        ax1.set_ylabel("Current (pA)", fontsize=9)
        ax1.legend(fontsize=7, loc="upper right")

        # ---- Na gating (row 2 — compact) ---------------------------
        ax2.clear()
        ax2.plot(t, m, label="m  (Na act)",   color=CP["m"], lw=0.9)
        ax2.plot(t, h, label="h  (Na inact)", color=CP["h"], lw=0.9)
        ax2.set_ylabel("Na", fontsize=8)
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.tick_params(labelsize=7)

        # ---- Kv3.1 gating (row 3 — compact) -----------------------
        ax3.clear()
        ax3.plot(t, n, label="n  (Kv3.1)",     color=CP["n"], lw=0.9)
        ax3.plot(t, p, label="p  (slow mod.)", color=CP["p"], lw=0.9)
        ax3.set_ylabel("Kv3.1", fontsize=8)
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=7, loc="upper right")
        ax3.tick_params(labelsize=7)

        # ---- K-LT gating (row 4 — compact) ------------------------
        ax4.clear()
        ax4.plot(t, l, label="l  (K-LT act)",   color=CP["l"], lw=0.9)
        ax4.plot(t, r, label="r  (K-LT inact)", color=CP["r"], lw=0.9)
        ax4.set_ylabel("K-LT", fontsize=8)
        ax4.set_ylim(-0.05, 1.05)
        ax4.legend(fontsize=7, loc="upper right")
        ax4.set_xlabel("Time (ms)", fontsize=9)
        ax4.tick_params(labelsize=7)

        self.fig.tight_layout(pad=2.5)
        self.canvas.draw()

        self.results_txt.delete("1.0", "end")
        self.results_txt.insert("end", "\n".join(analyze_ap(t, V)))

    def _err(self, msg):
        self.results_txt.delete("1.0", "end")
        self.results_txt.insert("end", f"ERROR:\n{msg}")
        print(f"[ERROR] {msg}")


# ==========================================================
if __name__ == "__main__":
    root = tk.Tk()
    MacicaGUI(root)
    root.mainloop()