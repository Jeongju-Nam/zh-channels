from pathlib import Path

import uproot
import awkward as ak
import numpy as np
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

PHOTON_PID = 22
HIGGS_PID = 25

PSEUDORAPIDITY_MAXIMUM = 2.5
PHOTON_PT_MINIMUM = 20

# A reconstructed photon counts as matched when it lies this close to a Higgs photon
PHOTON_MATCHING_DELTA_R = 0.1

# Events below this mass form the low-side tail under study
TAIL_MASS_MAXIMUM = 110


# Calculate the angular distance in the eta-phi plane
def delta_r(eta1, phi1, eta2, phi2):
    delta_eta = eta1 - eta2
    delta_phi = (phi1 - phi2 + np.pi)%(2*np.pi) - np.pi

    return np.sqrt(delta_eta**2 + delta_phi**2)


# Select the particles that a given mother decayed into
# The status code is unreliable here: a photon that reaches the detector
# is stored with status 1 rather than with a hard-process code
def select_decay_products(particle_pid, particle_m1, daughter_pid, mother_pid):
    mother = particle_pid[particle_m1]

    return (abs(particle_pid) == daughter_pid) & (mother == mother_pid)


# Format the fraction of a boolean subset, leaving the column blank when it is empty
def format_percentage(values, width = 12):
    if len(values) == 0:
        return f"{'-':>{width}}"

    return f"{100*np.mean(values):>{width - 1}.1f}%"


# Format the median of a subset, leaving the column blank when it is empty
def format_median(values, decimals, width = 12):
    if len(values) == 0:
        return f"{'-':>{width}}"

    return f"{np.median(values):>{width}.{decimals}f}"


# Construct an unweighted histogram
def make_histogram(values, bins, minimum, maximum):
    histogram = Hist.new.Regular(
        bins,
        minimum,
        maximum
    ).Int64()

    histogram.fill(values)

    return histogram


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


# Assign the two leading photons to the two Higgs photons
# Only two assignments are possible, so the one with the smaller total distance wins
def match_to_truth(
    leading_eta,
    leading_phi,
    subleading_eta,
    subleading_phi,
    truth_eta,
    truth_phi
):
    padded_eta = ak.to_numpy(ak.fill_none(ak.pad_none(truth_eta, 2, clip = True), np.nan))
    padded_phi = ak.to_numpy(ak.fill_none(ak.pad_none(truth_phi, 2, clip = True), np.nan))

    straight_leading = delta_r(leading_eta, leading_phi, padded_eta[:, 0], padded_phi[:, 0])
    straight_subleading = delta_r(subleading_eta, subleading_phi, padded_eta[:, 1], padded_phi[:, 1])
    crossed_leading = delta_r(leading_eta, leading_phi, padded_eta[:, 1], padded_phi[:, 1])
    crossed_subleading = delta_r(subleading_eta, subleading_phi, padded_eta[:, 0], padded_phi[:, 0])

    use_straight = (straight_leading + straight_subleading) <= (crossed_leading + crossed_subleading)

    leading_distance = np.where(use_straight, straight_leading, crossed_leading)
    subleading_distance = np.where(use_straight, straight_subleading, crossed_subleading)

    return (
        leading_distance < PHOTON_MATCHING_DELTA_R,
        subleading_distance < PHOTON_MATCHING_DELTA_R,
        leading_distance,
        subleading_distance
    )


def photon_truth_matching():
    tree = uproot.open(ZH_FILE)["Delphes"]

    particle_pid = tree["Particle.PID"].array()
    particle_m1 = tree["Particle.M1"].array()
    particle_eta = tree["Particle.Eta"].array()
    particle_phi = tree["Particle.Phi"].array()

    photon_pt = tree["Photon.PT"].array()
    photon_eta = tree["Photon.Eta"].array()
    photon_phi = tree["Photon.Phi"].array()
    photon_isolation = tree["Photon.IsolationVar"].array()

    is_higgs_photon = select_decay_products(particle_pid, particle_m1, PHOTON_PID, HIGGS_PID)

    has_two_photons = ak.num(photon_pt) >= 2

    pt = photon_pt[has_two_photons]
    eta = photon_eta[has_two_photons]
    phi = photon_phi[has_two_photons]
    isolation = photon_isolation[has_two_photons]

    truth_eta = particle_eta[is_higgs_photon][has_two_photons]
    truth_phi = particle_phi[is_higgs_photon][has_two_photons]

    leading_pt, subleading_pt = ak.to_numpy(pt[:, 0]), ak.to_numpy(pt[:, 1])
    leading_eta, subleading_eta = ak.to_numpy(eta[:, 0]), ak.to_numpy(eta[:, 1])
    leading_phi, subleading_phi = ak.to_numpy(phi[:, 0]), ak.to_numpy(phi[:, 1])
    leading_isolation = ak.to_numpy(isolation[:, 0])
    subleading_isolation = ak.to_numpy(isolation[:, 1])

    # Since photons are massless, E = pT*cosh(eta)
    px = leading_pt*np.cos(leading_phi) + subleading_pt*np.cos(subleading_phi)
    py = leading_pt*np.sin(leading_phi) + subleading_pt*np.sin(subleading_phi)
    pz = leading_pt*np.sinh(leading_eta) + subleading_pt*np.sinh(subleading_eta)
    energy = leading_pt*np.cosh(leading_eta) + subleading_pt*np.cosh(subleading_eta)

    all_masses = np.sqrt(np.maximum(energy**2 - (px**2 + py**2 + pz**2), 0))

    # The baseline selection of the tutorial, so that the tail under study is the one plotted
    pass_baseline = (
        (abs(leading_eta) < PSEUDORAPIDITY_MAXIMUM) &
        (abs(subleading_eta) < PSEUDORAPIDITY_MAXIMUM) &
        (leading_pt > PHOTON_PT_MINIMUM) &
        (subleading_pt > PHOTON_PT_MINIMUM)
    )

    leading_matched, subleading_matched, leading_distance, subleading_distance = match_to_truth(
        leading_eta,
        leading_phi,
        subleading_eta,
        subleading_phi,
        truth_eta,
        truth_phi
    )

    n_truth_photons = ak.to_numpy(ak.num(truth_eta))

    masses = all_masses[pass_baseline]
    leading_matched = leading_matched[pass_baseline]
    subleading_matched = subleading_matched[pass_baseline]
    leading_distance = leading_distance[pass_baseline]
    subleading_distance = subleading_distance[pass_baseline]
    leading_isolation = leading_isolation[pass_baseline]
    subleading_isolation = subleading_isolation[pass_baseline]
    leading_pt = leading_pt[pass_baseline]
    subleading_pt = subleading_pt[pass_baseline]

    both_matched = leading_matched & subleading_matched
    one_matched = leading_matched ^ subleading_matched
    none_matched = ~leading_matched & ~subleading_matched

    in_tail = masses < TAIL_MASS_MAXIMUM
    in_core = ~in_tail

    n_selected = len(masses)
    n_tail = int(np.sum(in_tail))
    n_core = int(np.sum(in_core))

    all_distances = np.concatenate([leading_distance, subleading_distance])

    print()
    print("Truth photons")
    print(f"{'Higgs photons per event':<26}: {n_truth_photons.mean():>9.2f}")
    print(f"{'events with exactly two':<26}: {int(np.sum(n_truth_photons == 2)):>9}")
    print(f"{'events with another count':<26}: {int(np.sum(n_truth_photons != 2)):>9}")

    print()
    print("Truth matching of the selected photon pair")
    print(f"{'selected pairs':<26}: {n_selected:>9}")
    print(f"{'both matched':<26}: {int(np.sum(both_matched)):>9} ({100*np.mean(both_matched):.1f}%)")
    print(f"{'exactly one matched':<26}: {int(np.sum(one_matched)):>9} ({100*np.mean(one_matched):.1f}%)")
    print(f"{'neither matched':<26}: {int(np.sum(none_matched)):>9} ({100*np.mean(none_matched):.1f}%)")
    print(f"{'median delta R':<26}: {np.median(all_distances):>9.4f}")

    print()
    print(f"Split at M = {TAIL_MASS_MAXIMUM} GeV")
    print(f"{'':<26}{'tail':>12}{'core':>12}")
    print(f"{'pairs':<26}{n_tail:>12}{n_core:>12}")
    print(
        f"{'both matched':<26}"
        f"{format_percentage(both_matched[in_tail])}"
        f"{format_percentage(both_matched[in_core])}"
    )
    print(
        f"{'exactly one matched':<26}"
        f"{format_percentage(one_matched[in_tail])}"
        f"{format_percentage(one_matched[in_core])}"
    )
    print(
        f"{'neither matched':<26}"
        f"{format_percentage(none_matched[in_tail])}"
        f"{format_percentage(none_matched[in_core])}"
    )

    print()
    print("Subleading photon properties")
    print(f"{'':<26}{'tail':>12}{'core':>12}")
    print(
        f"{'median pT [GeV]':<26}"
        f"{format_median(subleading_pt[in_tail], 2)}"
        f"{format_median(subleading_pt[in_core], 2)}"
    )
    print(
        f"{'median isolation':<26}"
        f"{format_median(subleading_isolation[in_tail], 4)}"
        f"{format_median(subleading_isolation[in_core], 4)}"
    )

    draw_histogram_pair(
        leading_distance,
        subleading_distance,
        "Leading photon",
        "Subleading photon",
        50,
        0,
        0.5,
        r"$\Delta R$(reconstructed, Higgs photon)",
        "Number of photons",
        "photon_truth_delta_r_raw_count_linear.png"
    )

    draw_histogram_pair(
        leading_distance,
        subleading_distance,
        "Leading photon",
        "Subleading photon",
        50,
        0,
        0.5,
        r"$\Delta R$(reconstructed, Higgs photon)",
        "Number of photons",
        "photon_truth_delta_r_raw_count_log.png",
        log_scale = True
    )

    # The tail can be empty in a small sample, and an empty histogram cannot be normalized
    if n_tail > 0:
        draw_histogram_pair(
            subleading_pt[in_tail],
            subleading_pt[in_core],
            rf"$M(\gamma\gamma) < {TAIL_MASS_MAXIMUM}$ GeV",
            rf"$M(\gamma\gamma) \geq {TAIL_MASS_MAXIMUM}$ GeV",
            50,
            0,
            100,
            r"Subleading photon $p_T$ [GeV]",
            r"Probability density [GeV$^{-1}$]",
            "subleading_photon_pt_tail_vs_core_normalized.png",
            density = True
        )

        draw_histogram_pair(
            subleading_isolation[in_tail],
            subleading_isolation[in_core],
            rf"$M(\gamma\gamma) < {TAIL_MASS_MAXIMUM}$ GeV",
            rf"$M(\gamma\gamma) \geq {TAIL_MASS_MAXIMUM}$ GeV",
            50,
            0,
            0.15,
            "Subleading photon isolation",
            "Probability density",
            "subleading_photon_isolation_tail_vs_core_normalized.png",
            density = True
        )

    draw_histogram_pair(
        masses[both_matched],
        masses[~both_matched],
        "Both photons matched",
        "At least one unmatched",
        50,
        40,
        140,
        r"$M(\gamma\gamma)$ [GeV]",
        "Number of selected events",
        "diphoton_mass_by_truth_matching_raw_count_linear.png"
    )

    draw_histogram_pair(
        masses[both_matched],
        masses[~both_matched],
        "Both photons matched",
        "At least one unmatched",
        50,
        40,
        140,
        r"$M(\gamma\gamma)$ [GeV]",
        "Number of selected events",
        "diphoton_mass_by_truth_matching_raw_count_log.png",
        log_scale = True
    )


if __name__ == "__main__":
    photon_truth_matching()
