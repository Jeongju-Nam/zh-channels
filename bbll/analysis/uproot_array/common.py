from pathlib import Path

import uproot
import awkward as ak
import numpy as np
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.special import voigt_profile
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

ZH_FILE = SAMPLE_DIRECTORY/f"ppTOzh_zTOll_hTObb_14TeV_{SAMPLE_SIZE}"/"Events"/"run_01"/"tag_1_delphes_events.root"
TTBAR_FILE = SAMPLE_DIRECTORY/f"ppTOtt_dileptonic_14TeV_{SAMPLE_SIZE}"/"Events"/"run_01"/"tag_1_delphes_events.root"

BOTTOM_PID = 5
ELECTRON_PID = 11
MUON_PID = 13
Z_PID = 23
HIGGS_PID = 25

# The jet clustering radius in the CMS Delphes card (ParameterR = 0.5),
# which is also what JetFlavorAssociation uses, so the same cone matches the jets
BOTTOM_MATCHING_DELTA_R = 0.5

# The leptons are measured directly, so a tighter criterion applies
LEPTON_MATCHING_DELTA_R = 0.1

Z_MASS_PDG = 91.1876
Z_WIDTH_PDG = 2.4952

# The width the generator was run with, taken from the run banner (DECAY 23)
Z_WIDTH_GENERATOR = 2.441404

# The window isolates the resonance core; a wider one lets the tails inflate the width
Z_FIT_MINIMUM = 85
Z_FIT_MAXIMUM = 97

MATCHING_METHODS = ["all_pairs", "nearest", "greedy", "optimal"]


# Count the values outside the plotting range
def count_outside(values, minimum, maximum):
    return int(np.sum((values < minimum) | (values > maximum)))


# Calculate the invariant mass of the two leading objects in each event
def invariant_mass(pt, eta, phi, mass):
    px = pt[:, 0]*np.cos(phi[:, 0]) + pt[:, 1]*np.cos(phi[:, 1])
    py = pt[:, 0]*np.sin(phi[:, 0]) + pt[:, 1]*np.sin(phi[:, 1])
    pz = pt[:, 0]*np.sinh(eta[:, 0]) + pt[:, 1]*np.sinh(eta[:, 1])

    leading_energy = np.sqrt((pt[:, 0]*np.cosh(eta[:, 0]))**2 + mass[:, 0]**2)
    subleading_energy = np.sqrt((pt[:, 1]*np.cosh(eta[:, 1]))**2 + mass[:, 1]**2)
    energy = leading_energy + subleading_energy

    return np.sqrt(np.maximum(ak.to_numpy(energy**2 - (px**2 + py**2 + pz**2)), 0))


# Select the particles that a given mother decayed into
# The status code is unreliable here: a photon or a lepton that reaches the detector
# is stored with status 1 rather than with a hard-process code
def select_decay_products(particle_pid, particle_m1, daughter_pid, mother_pid):
    mother = particle_pid[particle_m1]

    return (abs(particle_pid) == daughter_pid) & (mother == mother_pid)


# Merge the electron and muon collections into one pT-ordered lepton collection
def merge_leptons(electron_pt, electron_eta, electron_phi, muon_pt, muon_eta, muon_phi):
    lepton_pt = ak.concatenate([electron_pt, muon_pt], axis = 1)
    lepton_eta = ak.concatenate([electron_eta, muon_eta], axis = 1)
    lepton_phi = ak.concatenate([electron_phi, muon_phi], axis = 1)

    # Concatenation breaks the pT ordering, so the merged collection is sorted again
    order = ak.argsort(lepton_pt, ascending = False)

    return lepton_pt[order], lepton_eta[order], lepton_phi[order]


# Calculate the angular distance in the eta-phi plane
def delta_r(eta1, phi1, eta2, phi2):
    delta_eta = eta1 - eta2
    delta_phi = (phi1 - phi2 + np.pi)%(2*np.pi) - np.pi

    return np.sqrt(delta_eta**2 + delta_phi**2)


# Build the distance matrix between the partons and the reconstructed objects of one event
def build_distance_matrix(parton_eta, parton_phi, reconstructed_eta, reconstructed_phi):
    matrix = np.zeros((len(parton_eta), len(reconstructed_eta)))

    for j in range(len(parton_eta)):
        matrix[j] = delta_r(
            parton_eta[j],
            parton_phi[j],
            reconstructed_eta,
            reconstructed_phi
        )

    return matrix


# Keep every parton-object distance, which is what the original tutorial plots
def assign_all_pairs(matrix):
    rows, columns = np.meshgrid(
        np.arange(matrix.shape[0]),
        np.arange(matrix.shape[1]),
        indexing = "ij"
    )

    return matrix.flatten(), columns.flatten()


# Let each parton take its closest object, so one object may serve several partons
def assign_nearest(matrix):
    return matrix.min(axis = 1), matrix.argmin(axis = 1)


# Confirm the closest pair first, then remove its parton and object, and repeat
def assign_greedy(matrix):
    remaining = matrix.copy()
    distances = []
    columns = []

    while np.isfinite(remaining).any():
        flat_index = np.argmin(remaining)
        row, column = np.unravel_index(flat_index, remaining.shape)

        distances.append(remaining[row, column])
        columns.append(column)

        remaining[row, :] = np.inf
        remaining[:, column] = np.inf

    return np.array(distances), np.array(columns)


# Choose the assignment whose total distance is smallest
def assign_optimal(matrix):
    rows, columns = linear_sum_assignment(matrix)

    return matrix[rows, columns], columns


# Collect the matching distances of every event with the chosen method
# A flag array such as Jet.BTag can be passed so that the matched objects
# carry their tag alongside the distance
def match_to_reconstructed(
    parton_eta,
    parton_phi,
    reconstructed_eta,
    reconstructed_phi,
    method,
    reconstructed_flag = None
):
    assign = {
        "all_pairs": assign_all_pairs,
        "nearest": assign_nearest,
        "greedy": assign_greedy,
        "optimal": assign_optimal
    }[method]

    distances = []
    flags = []
    n_without_candidate = 0
    n_duplicate_events = 0
    n_matchable_events = 0

    for i in range(len(parton_eta)):
        if len(reconstructed_eta[i]) == 0:
            n_without_candidate += len(parton_eta[i])
            continue

        matrix = build_distance_matrix(
            ak.to_numpy(parton_eta[i]),
            ak.to_numpy(parton_phi[i]),
            ak.to_numpy(reconstructed_eta[i]),
            ak.to_numpy(reconstructed_phi[i])
        )

        event_distances, event_columns = assign(matrix)
        distances.append(event_distances)

        if reconstructed_flag is not None:
            event_flag = ak.to_numpy(reconstructed_flag[i])
            flags.append(event_flag[event_columns])

        # Count how often the nearest-object rule hands one object to several partons
        if len(parton_eta[i]) >= 2:
            n_matchable_events += 1

            chosen = matrix.argmin(axis = 1)
            if len(set(chosen)) < len(chosen):
                n_duplicate_events += 1

    return (
        np.concatenate(distances) if distances else np.array([]),
        np.concatenate(flags) if flags else np.array([]),
        n_without_candidate,
        n_duplicate_events,
        n_matchable_events
    )


# Extract the dilepton and bbbar invariant masses from one sample
def extract_masses(input_file):
    tree = uproot.open(input_file)["Delphes"]

    electron_pt = tree["Electron.PT"].array()
    electron_eta = tree["Electron.Eta"].array()
    electron_phi = tree["Electron.Phi"].array()

    muon_pt = tree["Muon.PT"].array()
    muon_eta = tree["Muon.Eta"].array()
    muon_phi = tree["Muon.Phi"].array()

    jet_pt = tree["Jet.PT"].array()
    jet_eta = tree["Jet.Eta"].array()
    jet_phi = tree["Jet.Phi"].array()
    jet_mass = tree["Jet.Mass"].array()
    jet_btag = tree["Jet.BTag"].array()

    # The electrons and muons are merged so that the mixed-flavour pairs are kept
    lepton_pt, lepton_eta, lepton_phi = merge_leptons(
        electron_pt,
        electron_eta,
        electron_phi,
        muon_pt,
        muon_eta,
        muon_phi
    )

    has_two_leptons = ak.num(lepton_pt) >= 2
    lepton_pt = lepton_pt[has_two_leptons]
    lepton_eta = lepton_eta[has_two_leptons]
    lepton_phi = lepton_phi[has_two_leptons]

    dilepton_mass = invariant_mass(
        lepton_pt,
        lepton_eta,
        lepton_phi,
        ak.zeros_like(lepton_pt)
    )

    # The two highest-pT b-tagged jets
    is_btagged = jet_btag == 1
    bottom_pt = jet_pt[is_btagged]
    bottom_eta = jet_eta[is_btagged]
    bottom_phi = jet_phi[is_btagged]
    bottom_mass = jet_mass[is_btagged]

    has_two_bottom_jets = ak.num(bottom_pt) >= 2
    bottom_pt = bottom_pt[has_two_bottom_jets]
    bottom_eta = bottom_eta[has_two_bottom_jets]
    bottom_phi = bottom_phi[has_two_bottom_jets]
    bottom_mass = bottom_mass[has_two_bottom_jets]

    bbbar_mass = invariant_mass(bottom_pt, bottom_eta, bottom_phi, bottom_mass)

    return tree.num_entries, dilepton_mass, bbbar_mass


# Evaluate a Breit-Wigner line shape
def breit_wigner(mass, amplitude, peak, width):
    return amplitude/((mass - peak)**2 + (width/2)**2)


# Fit a Breit-Wigner to the parton-level Z line shape
def fit_z_width(masses, minimum = Z_FIT_MINIMUM, maximum = Z_FIT_MAXIMUM, bins = 40):
    counts, edges = np.histogram(masses, bins = bins, range = (minimum, maximum))
    centres = 0.5*(edges[1:] + edges[:-1])

    parameters, covariance = curve_fit(
        breit_wigner,
        centres,
        counts,
        p0 = [counts.max()*1.5, Z_MASS_PDG, Z_WIDTH_PDG],
        sigma = np.sqrt(np.maximum(counts, 1)),
        absolute_sigma = True
    )

    return parameters, np.sqrt(np.diag(covariance))


# Build a Voigt profile with the natural width held fixed
def make_voigt(natural_width):
    def voigt(mass, amplitude, peak, resolution):
        return amplitude*voigt_profile(mass - peak, resolution, natural_width/2)

    return voigt


# Fit a Voigt profile to the reconstructed Z line shape
def fit_z_resolution(
    masses,
    natural_width,
    minimum = Z_FIT_MINIMUM,
    maximum = Z_FIT_MAXIMUM,
    bins = 40
):
    counts, edges = np.histogram(masses, bins = bins, range = (minimum, maximum))
    centres = 0.5*(edges[1:] + edges[:-1])

    voigt = make_voigt(natural_width)

    parameters, covariance = curve_fit(
        voigt,
        centres,
        counts,
        p0 = [counts.max(), Z_MASS_PDG, 2],
        sigma = np.sqrt(np.maximum(counts, 1)),
        absolute_sigma = True
    )

    return parameters, np.sqrt(np.diag(covariance)), voigt


# Construct an unweighted histogram
def make_histogram(values, bins, minimum, maximum):
    histogram = Hist.new.Regular(
        bins,
        minimum,
        maximum
    ).Int64()

    histogram.fill(values)

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
    density = False,
    log_scale = False
):
    histogram = make_histogram(
        values,
        bins,
        minimum,
        maximum
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


# Draw two overlaid histograms
def draw_histogram_pair(
    values_1,
    values_2,
    label_1,
    label_2,
    bins,
    minimum,
    maximum,
    xlabel,
    ylabel,
    filename,
    density = False,
    log_scale = False
):
    histogram_1 = make_histogram(
        values_1,
        bins,
        minimum,
        maximum
    )

    histogram_2 = make_histogram(
        values_2,
        bins,
        minimum,
        maximum
    )

    fig, ax = plt.subplots(figsize = (8, 6))

    histplot(
        histogram_1,
        ax = ax,
        histtype = "barstep",
        yerr = True,
        density = density,
        linewidth = 1.5,
        label = label_1
    )

    histplot(
        histogram_2,
        ax = ax,
        histtype = "barstep",
        yerr = True,
        density = density,
        linewidth = 1.5,
        linestyle = "--",
        label = label_2
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon = False)
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
