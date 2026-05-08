import os
import sys

# ==========================================================
# DYLD_LIBRARY_PATH must be set BEFORE the process starts.
# Re-exec with correct env if needed.
# ==========================================================
NESTML_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hh_custom.nestml")
MODULE_TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hh_custom_module")
MODULE_NAME   = "hh_custom_module"
NEURON_MODEL  = "hh_custom_nestml"

_REEXEC_FLAG = "__SCA13_REEXEC__"

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
    print("[NESTML] Compiling hh_custom.nestml ...")

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
# GUI
# ==========================================================
class SCAGUI:
    def __init__(self, root):
        self.root = root
        root.title("SCA13 Neuron Tuner — KCNC3 Loss-of-Function")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        # ---- Scrollable left panel ----------------------------------
        left_outer = ttk.Frame(root)
        left_outer.grid(row=0, column=0, sticky="ns")

        cv = tk.Canvas(left_outer, width=330, highlightthickness=0)
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
        ttk.Label(ctrl, text="SCA13 Neuron Tuner",
                  font=("", 11, "bold")).pack(pady=(0, 4))
        self.preset_var = tk.StringVar(value="MNTB Physio")
        ttk.OptionMenu(ctrl, self.preset_var, "MNTB Physio",
                       *PRESETS.keys(),
                       command=self.load_preset).pack(fill="x", pady=4)

        # ---- Parameter groups ---------------------------------------
        self.vars = {}
        # Store defaults for reset: {param: default_value}
        self._defaults = {}

        for group_name, params in PARAM_GROUPS.items():
            grp = ttk.LabelFrame(ctrl, text=group_name, padding=5)
            grp.pack(fill="x", pady=3)

            for param, default, lo, hi, res, tooltip in params:
                self.vars[param] = tk.DoubleVar(value=default)
                self._defaults[param] = default

                row = ttk.Frame(grp)
                row.pack(fill="x", pady=2)

                lbl = ttk.Label(row, text=param, width=10, anchor="w")
                lbl.pack(side="left")
                lbl.bind("<Enter>", lambda e, t=tooltip: self.tooltip.config(text=t))
                lbl.bind("<Leave>", lambda e: self.tooltip.config(text=""))

                ent = ttk.Entry(row, textvariable=self.vars[param], width=8)
                ent.pack(side="right")
                ent.bind("<Return>", lambda e: self.run())

                ttk.Scale(row, from_=lo, to=hi,
                          variable=self.vars[param], orient="horizontal",
                          command=lambda x, p=param, r=res: self._snap(p, r)
                          ).pack(side="left", fill="x", expand=True)

        # ---- Stimulation -------------------------------------------
        stim = ttk.LabelFrame(ctrl, text="Stimulation", padding=5)
        stim.pack(fill="x", pady=4)
        self.dc_var = tk.DoubleVar(value=200.0)
        self._dc_default = 200.0
        ttk.Label(stim, text="DC Injection (pA)").pack()
        ttk.Scale(stim, from_=0, to=2000, variable=self.dc_var,
                  command=lambda x: self.run()).pack(fill="x")
        ttk.Entry(stim, textvariable=self.dc_var, width=8).pack()

        # ---- AP analysis --------------------------------------------
        rf = ttk.LabelFrame(ctrl, text="AP Analysis", padding=5)
        rf.pack(fill="both", expand=True, pady=4)
        self.results_txt = tk.Text(rf, height=12, width=34, font=("Courier", 8))
        self.results_txt.pack(fill="both", expand=True)

        # ---- Run / Reset buttons ------------------------------------
        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(fill="x", pady=6)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=1)
        ttk.Button(btn_frame, text="▶  Run",   command=self.run).grid(row=0, column=0, sticky="ew", padx=(0, 2))
        ttk.Button(btn_frame, text="↺  Reset", command=self.reset).grid(row=0, column=1, sticky="ew", padx=(2, 0))

        # ---- Tooltip bar -------------------------------------------
        self.tooltip = ttk.Label(root, text="", foreground="gray",
                                 font=("", 8, "italic"))
        self.tooltip.grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        # ---- Plots --------------------------------------------------
        self.fig, self.axes = plt.subplots(
            3, 1, figsize=(7, 9), sharex=True,
            gridspec_kw={"height_ratios": [3, 2, 2]}
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

    def load_preset(self, name):
        for k, v in PRESETS[name].items():
            if k in self.vars:
                self.vars[k].set(v)
        self.run()

    def reset(self):
        """Restore all parameters to their default values and re-run."""
        for param, default in self._defaults.items():
            self.vars[param].set(default)
        self.dc_var.set(self._dc_default)
        self.preset_var.set("MNTB Physio")
        self.run()

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

        dc = nest.Create("dc_generator",
                         params={"amplitude": self.dc_var.get(),
                                 "start": 5.0, "stop": 45.0})
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
        t, V = ev["times"], ev["V_m"]
        n, m, h = ev["n"], ev["m"], ev["h"]

        gK, gNa = self.vars["g_K"].get(), self.vars["g_Na"].get()
        EK, ENa = self.vars["E_K"].get(), self.vars["E_Na"].get()
        I_K  = gK  * n**4     * (V - EK)  / 1000.0
        I_Na = gNa * m**3 * h * (V - ENa) / 1000.0

        ax0, ax1, ax2 = self.axes

        ax0.clear()
        ax0.plot(t, V, color=CP["Vm"], lw=1.0)
        ax0.set_ylabel("V_m (mV)")
        ax0.set_title(f"{self.preset_var.get()}   |   DC = {self.dc_var.get():.0f} pA",
                      fontsize=9)
        ax0.axhline(0, color="gray", lw=0.4, ls="--")

        ax1.clear()
        ax1.plot(t, n, label="n (K act)",    color=CP["K"],  lw=1.0)
        ax1.plot(t, m, label="m (Na act)",   color=CP["Na"], lw=1.0)
        ax1.plot(t, h, label="h (Na inact)", color=CP["h"],  lw=1.0)
        ax1.set_ylabel("Gating variables")
        ax1.legend(fontsize=7, loc="upper right")

        ax2.clear()
        ax2.plot(t, I_K,  label="I_K (nA)",  color=CP["K"],  lw=1.0)
        ax2.plot(t, I_Na, label="I_Na (nA)", color=CP["Na"], lw=1.0)
        ax2.axhline(0, color="gray", lw=0.4, ls="--")
        ax2.set_ylabel("Current (nA)")
        ax2.set_xlabel("Time (ms)")
        ax2.legend(fontsize=7, loc="upper right")

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
    SCAGUI(root)
    root.mainloop()