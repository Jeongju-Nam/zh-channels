from pathlib import Path

import uproot
import awkward as ak
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hist import Hist
from hist.plot import histplot

SAMPLE_SIZE = "100k"

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent.parent

SAMPLE_DIRECTORY = PROJECT_DIRECTORY/"runs"
FIGURE_DIRECTORY = PROJECT_DIRECTORY/"figures"/"python"/SAMPLE_SIZE
FIGURE_DIRECTORY.mkdir(parents = True, exist_ok = True)

ZH_FILE = SAMPLE_DIRECTORY/f"ppTOzh_zTOvv_hTOaa_14TeV_{SAMPLE_SIZE}"/"Events"/"run_01"/"tag_1_delphes_events.root"

HIGGS_MASS_PDG = 125.0

# The half-window used to isolate the Higgs peak core from the combinatorial tail
HIGGS_FIT_HALF_WINDOW = 8

CROSS_SECTION_TIMES_BRANCHING = 0.0004
INTEGRATED_LUMINOSITY = 300

PSEUDORAPIDITY_MAXIMUM = 2.5

# The fixed threshold of the tutorial selection
PHOTON_PT_MINIMUM = 20

# A tighter isolation than the one the CMS Delphes card already applies
PHOTON_ISOLATION_MAXIMUM = 0.05

# The CMS H to diphoton analysis scales the thresholds with the diphoton mass
LEADING_PT_MASS_FRACTION = 1/3
SUBLEADING_PT_MASS_FRACTION = 1/4

# Each mode is a label with the two switches it turns on
SELECTION_MODES = [
    ("baseline", False, False),
    ("isolation", True, False),
    ("proportional", False, True),
    ("isolation_and_proportional", True, True)
]


# Count the values outside the plotting range
def count_outside(values, minimum, maximum):
    return int(np.sum((values < minimum) | (values > maximum)))


# Evaluate a Gaussian line shape
def gaussian(mass, amplitude, peak, resolution):
    return amplitude*np.exp(-0.5*((mass - peak)/resolution)**2)


# Fit a Gaussian to the core of the diphoton peak
def fit_higgs_peak(masses, half_window = HIGGS_FIT_HALF_WINDOW, bins = 40):
    core = masses[np.abs(masses - HIGGS_MASS_PDG) < half_window]

    counts, edges = np.histogram(
        core,
        bins = bins,
        range = (HIGGS_MASS_PDG - half_window, HIGGS_MASS_PDG + half_window)
    )

    centres = 0.5*(edges[1:] + edges[:-1])

    parameters, covariance = curve_fit(
        gaussian,
        centres,
        counts,
        p0 = [counts.max(), HIGGS_MASS_PDG, 2.5],
        sigma = np.sqrt(np.maximum(counts, 1)),
        absolute_sigma = True
    )

    return parameters, np.sqrt(np.diag(covariance)), len(core)


# Construct a histogram, weighted if weights are given
def make_histogram(
    values,
    bins,
    minimum,
    maximum,
    weights = None
):
    if weights is None:
        histogram = Hist.new.Regular(
            bins,
            minimum,
            maximum
        ).Int64()

        histogram.fill(values)
    else:
        histogram = Hist.new.Regular(
            bins,
            minimum,
            maximum
        ).Weight()

        histogram.fill(values, weight = weights)

    return histogram


# Draw one histogram
def draw_histogram(
    values,
    bins,
    minimum,
    maximum,
    xlabel,
    ylabel,
    filename,
    weights = None,
    density = False,
    log_scale = False
):
    histogram = make_histogram(
        values,
        bins,
        minimum,
        maximum,
        weights
    )

    fig, ax = plt.subplots(figsize = (8, 6))

    histplot(
        histogram,
        ax = ax,
        histtype = "barstep",
        yerr = True,
        density = density,
        linewidth = 1.5
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    if log_scale:
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY/filename, bbox_inches = "tight")
    plt.close(fig)


# Draw a histogram together with its fitted curve
def draw_histogram_with_fit(
    values,
    fit_function,
    parameters,
    bins,
    minimum,
    maximum,
    xlabel,
    ylabel,
    label,
    filename
):
    counts, edges = np.histogram(values, bins = bins, range = (minimum, maximum))
    centres = 0.5*(edges[1:] + edges[:-1])

    fig, ax = plt.subplots(figsize = (8, 6))

    ax.errorbar(
        centres,
        counts,
        yerr = np.sqrt(counts),
        fmt = "o",
        markersize = 3,
        capsize = 3,
        label = "Selected events"
    )

    curve_x = np.linspace(minimum, maximum, 400)
    ax.plot(curve_x, fit_function(curve_x, *parameters), linewidth = 1.5, label = label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon = False)
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY/filename, bbox_inches = "tight")
    plt.close(fig)


# Read the photon kinematics and compute the diphoton mass before any selection
def read_photon_candidates(tree):
    photon_pt = tree["Photon.PT"].array()
    photon_eta = tree["Photon.Eta"].array()
    photon_phi = tree["Photon.Phi"].array()
    photon_isolation = tree["Photon.IsolationVar"].array()
    met = tree["MissingET.MET"].array()

    has_two_photons = ak.num(photon_pt) >= 2

    pt = photon_pt[has_two_photons]
    eta = photon_eta[has_two_photons]
    phi = photon_phi[has_two_photons]
    isolation = photon_isolation[has_two_photons]
    selected_events_met = met[has_two_photons]

    leading_pt, subleading_pt = pt[:, 0], pt[:, 1]
    leading_eta, subleading_eta = eta[:, 0], eta[:, 1]
    leading_phi, subleading_phi = phi[:, 0], phi[:, 1]
    leading_isolation, subleading_isolation = isolation[:, 0], isolation[:, 1]

    # Since photons are massless, E = pT*cosh(eta)
    px = leading_pt*np.cos(leading_phi) + subleading_pt*np.cos(subleading_phi)
    py = leading_pt*np.sin(leading_phi) + subleading_pt*np.sin(subleading_phi)
    pz = leading_pt*np.sinh(leading_eta) + subleading_pt*np.sinh(subleading_eta)
    energy = leading_pt*np.cosh(leading_eta) + subleading_pt*np.cosh(subleading_eta)

    mass_squared = energy**2 - (px**2 + py**2 + pz**2)

    return {
        "n_total_events": tree.num_entries,
        "n_has_two_photons": int(ak.sum(has_two_photons)),
        "leading_pt": ak.to_numpy(leading_pt),
        "subleading_pt": ak.to_numpy(subleading_pt),
        "leading_eta": ak.to_numpy(leading_eta),
        "subleading_eta": ak.to_numpy(subleading_eta),
        "leading_isolation": ak.to_numpy(leading_isolation),
        "subleading_isolation": ak.to_numpy(subleading_isolation),
        "masses": np.sqrt(np.maximum(ak.to_numpy(mass_squared), 0)),
        "met": ak.to_numpy(ak.firsts(selected_events_met))
    }


# Build the selection masks of one mode and return what it keeps
# The cuts follow the usual order: acceptance, object identification, kinematics
def apply_selection(candidates, use_isolation, use_proportional_pt):
    all_masses = candidates["masses"]

    pass_pseudorapidity = (
        (abs(candidates["leading_eta"]) < PSEUDORAPIDITY_MAXIMUM) &
        (abs(candidates["subleading_eta"]) < PSEUDORAPIDITY_MAXIMUM)
    )

    if use_isolation:
        pass_isolation = (
            (candidates["leading_isolation"] < PHOTON_ISOLATION_MAXIMUM) &
            (candidates["subleading_isolation"] < PHOTON_ISOLATION_MAXIMUM)
        )
    else:
        pass_isolation = np.ones(len(all_masses), dtype = bool)

    if use_proportional_pt:
        pass_transverse_momentum = (
            (candidates["leading_pt"] > all_masses*LEADING_PT_MASS_FRACTION) &
            (candidates["subleading_pt"] > all_masses*SUBLEADING_PT_MASS_FRACTION)
        )
    else:
        pass_transverse_momentum = (
            (candidates["leading_pt"] > PHOTON_PT_MINIMUM) &
            (candidates["subleading_pt"] > PHOTON_PT_MINIMUM)
        )

    pass_all_cuts = pass_pseudorapidity & pass_isolation & pass_transverse_momentum

    cut_flow = {
        "total": candidates["n_total_events"],
        "two_photons": candidates["n_has_two_photons"],
        "pseudorapidity": int(np.sum(pass_pseudorapidity)),
        "isolation": int(np.sum(pass_pseudorapidity & pass_isolation)),
        "transverse_momentum": int(np.sum(pass_all_cuts))
    }

    return cut_flow, all_masses[pass_all_cuts], candidates["met"][pass_all_cuts]


def zh_diphoton_yield():
    tree = uproot.open(ZH_FILE)["Delphes"]
    candidates = read_photon_candidates(tree)

    mass_minimum = 40
    mass_maximum = 140

    met_minimum = 0
    met_maximum = 300

    weighted_label = f"Expected events at {INTEGRATED_LUMINOSITY} /fb"

    results = {}

    for label, use_isolation, use_proportional_pt in SELECTION_MODES:
        cut_flow, masses, met = apply_selection(candidates, use_isolation, use_proportional_pt)

        event_weight = (CROSS_SECTION_TIMES_BRANCHING*INTEGRATED_LUMINOSITY*1e3
                        /cut_flow["total"])

        n_selected = cut_flow["transverse_momentum"]
        fit_parameters, fit_errors, n_core = fit_higgs_peak(masses)

        results[label] = {
            "cut_flow": cut_flow,
            "masses": masses,
            "met": met,
            "event_weight": event_weight,
            "expected_yield": n_selected*event_weight,
            "yield_uncertainty": np.sqrt(n_selected)*event_weight,
            "fit_parameters": fit_parameters,
            "fit_errors": fit_errors,
            "n_core": n_core,
            "n_below_core": int(np.sum(masses < HIGGS_MASS_PDG - HIGGS_FIT_HALF_WINDOW))
        }

        draw_histogram(
            masses,
            50,
            mass_minimum,
            mass_maximum,
            r"$M(\gamma\gamma)$ [GeV]",
            weighted_label,
            f"zh_diphoton_mass_{label}_weighted_linear.png",
            weights = np.full_like(masses, event_weight)
        )

        draw_histogram(
            masses,
            50,
            mass_minimum,
            mass_maximum,
            r"$M(\gamma\gamma)$ [GeV]",
            weighted_label,
            f"zh_diphoton_mass_{label}_weighted_log.png",
            weights = np.full_like(masses, event_weight),
            log_scale = True
        )

        draw_histogram_with_fit(
            masses,
            gaussian,
            fit_parameters,
            40,
            HIGGS_MASS_PDG - HIGGS_FIT_HALF_WINDOW,
            HIGGS_MASS_PDG + HIGGS_FIT_HALF_WINDOW,
            r"$M(\gamma\gamma)$ [GeV]",
            "Number of selected events",
            rf"Gaussian, $\sigma$ = {abs(fit_parameters[2]):.2f} GeV",
            f"zh_diphoton_mass_{label}_raw_count_fit.png"
        )

        draw_histogram(
            met,
            50,
            met_minimum,
            met_maximum,
            "MET [GeV]",
            weighted_label,
            f"zh_missing_transverse_energy_{label}_weighted_linear.png",
            weights = np.full_like(met, event_weight)
        )

        draw_histogram(
            met,
            50,
            met_minimum,
            met_maximum,
            "MET [GeV]",
            weighted_label,
            f"zh_missing_transverse_energy_{label}_weighted_log.png",
            weights = np.full_like(met, event_weight),
            log_scale = True
        )

    labels = [label for label, _, _ in SELECTION_MODES]

    print()
    print("Selection modes")
    print(f"{'baseline':<28}: fixed pT > {PHOTON_PT_MINIMUM} GeV")
    print(f"{'isolation':<28}: isolation < {PHOTON_ISOLATION_MAXIMUM}")
    print(f"{'proportional':<28}: pT > M/3 and M/4")
    print(f"{'isolation_and_proportional':<28}: both of the above")

    print()
    print("Cut flow")
    print(f"{'':<28}{'total':>8}{'2 photons':>11}{'eta':>7}{'isolated':>10}{'pT':>7}")
    for label in labels:
        cut_flow = results[label]["cut_flow"]

        print(
            f"{label:<28}"
            f"{cut_flow['total']:>8}"
            f"{cut_flow['two_photons']:>11}"
            f"{cut_flow['pseudorapidity']:>7}"
            f"{cut_flow['isolation']:>10}"
            f"{cut_flow['transverse_momentum']:>7}"
        )

    print()
    print(f"Yield at {INTEGRATED_LUMINOSITY} /fb, with the MC statistical uncertainty")
    print(f"{'':<28}{'yield':>10}{'uncertainty':>14}")
    for label in labels:
        result = results[label]

        print(
            f"{label:<28}"
            f"{result['expected_yield']:>10.2f}"
            f"{result['yield_uncertainty']:>14.2f}"
        )

    print()
    print("Diphoton mass of the selected events")
    print(f"{'':<28}{'mean':>10}{'standard deviation':>20}{'below core':>13}")
    for label in labels:
        result = results[label]

        print(
            f"{label:<28}"
            f"{result['masses'].mean():>10.3f}"
            f"{result['masses'].std():>20.3f}"
            f"{result['n_below_core']:>13}"
        )

    print()
    print(f"Gaussian fit to the core, |M - {HIGGS_MASS_PDG}| < {HIGGS_FIT_HALF_WINDOW} GeV")
    print(f"{'':<28}{'entries':>9}{'peak':>10}{'error':>9}{'resolution':>12}{'error':>9}")
    for label in labels:
        result = results[label]

        print(
            f"{label:<28}"
            f"{result['n_core']:>9}"
            f"{result['fit_parameters'][1]:>10.3f}"
            f"{result['fit_errors'][1]:>9.3f}"
            f"{abs(result['fit_parameters'][2]):>12.3f}"
            f"{result['fit_errors'][2]:>9.3f}"
        )

    print()
    print("Entries outside plot range")
    print(f"{'':<28}{'diphoton mass':>16}{'MET':>8}")
    for label in labels:
        result = results[label]

        print(
            f"{label:<28}"
            f"{count_outside(result['masses'], mass_minimum, mass_maximum):>16}"
            f"{count_outside(result['met'], met_minimum, met_maximum):>8}"
        )


if __name__ == "__main__":
    zh_diphoton_yield()
