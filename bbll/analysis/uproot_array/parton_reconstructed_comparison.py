import uproot
import awkward as ak
import numpy as np

from common import (
    BOTTOM_MATCHING_DELTA_R,
    BOTTOM_PID,
    ELECTRON_PID,
    HIGGS_PID,
    LEPTON_MATCHING_DELTA_R,
    MATCHING_METHODS,
    MUON_PID,
    ZH_FILE,
    Z_FIT_MAXIMUM,
    Z_FIT_MINIMUM,
    Z_MASS_PDG,
    Z_PID,
    Z_WIDTH_GENERATOR,
    Z_WIDTH_PDG,
    breit_wigner,
    count_outside,
    draw_histogram,
    draw_histogram_pair,
    draw_histogram_with_fit,
    fit_z_resolution,
    fit_z_width,
    invariant_mass,
    match_to_reconstructed,
    merge_leptons,
    select_decay_products
)


def parton_reconstructed_comparison():
    tree = uproot.open(ZH_FILE)["Delphes"]

    particle_pid = tree["Particle.PID"].array()
    particle_m1 = tree["Particle.M1"].array()
    particle_pt = tree["Particle.PT"].array()
    particle_eta = tree["Particle.Eta"].array()
    particle_phi = tree["Particle.Phi"].array()

    # Particle.Mass is stored in MeV while PT and Eta are in GeV
    particle_mass = tree["Particle.Mass"].array()/1000

    jet_pt = tree["Jet.PT"].array()
    jet_eta = tree["Jet.Eta"].array()
    jet_phi = tree["Jet.Phi"].array()
    jet_mass = tree["Jet.Mass"].array()
    jet_btag = tree["Jet.BTag"].array()

    electron_pt = tree["Electron.PT"].array()
    electron_eta = tree["Electron.Eta"].array()
    electron_phi = tree["Electron.Phi"].array()

    muon_pt = tree["Muon.PT"].array()
    muon_eta = tree["Muon.Eta"].array()
    muon_phi = tree["Muon.Phi"].array()

    is_parton_bottom = select_decay_products(particle_pid, particle_m1, BOTTOM_PID, HIGGS_PID)
    is_parton_electron = select_decay_products(particle_pid, particle_m1, ELECTRON_PID, Z_PID)
    is_parton_muon = select_decay_products(particle_pid, particle_m1, MUON_PID, Z_PID)
    is_parton_lepton = is_parton_electron | is_parton_muon

    # Parton-level objects
    parton_bottom_pt = particle_pt[is_parton_bottom]
    parton_bottom_eta = particle_eta[is_parton_bottom]
    parton_bottom_phi = particle_phi[is_parton_bottom]
    parton_bottom_mass = particle_mass[is_parton_bottom]

    has_two_parton_bottoms = ak.num(parton_bottom_pt) >= 2
    parton_bbbar_mass = invariant_mass(
        parton_bottom_pt[has_two_parton_bottoms],
        parton_bottom_eta[has_two_parton_bottoms],
        parton_bottom_phi[has_two_parton_bottoms],
        parton_bottom_mass[has_two_parton_bottoms]
    )

    parton_lepton_pt = particle_pt[is_parton_lepton]
    parton_lepton_eta = particle_eta[is_parton_lepton]
    parton_lepton_phi = particle_phi[is_parton_lepton]
    parton_lepton_mass = particle_mass[is_parton_lepton]

    has_two_parton_leptons = ak.num(parton_lepton_pt) >= 2
    parton_dilepton_mass = invariant_mass(
        parton_lepton_pt[has_two_parton_leptons],
        parton_lepton_eta[has_two_parton_leptons],
        parton_lepton_phi[has_two_parton_leptons],
        parton_lepton_mass[has_two_parton_leptons]
    )

    # Reconstructed b-jets, used for the invariant mass
    is_btagged = jet_btag == 1
    reconstructed_bottom_pt = jet_pt[is_btagged]
    reconstructed_bottom_eta = jet_eta[is_btagged]
    reconstructed_bottom_phi = jet_phi[is_btagged]
    reconstructed_bottom_mass = jet_mass[is_btagged]

    has_two_bottom_jets = ak.num(reconstructed_bottom_pt) >= 2
    reconstructed_bbbar_mass = invariant_mass(
        reconstructed_bottom_pt[has_two_bottom_jets],
        reconstructed_bottom_eta[has_two_bottom_jets],
        reconstructed_bottom_phi[has_two_bottom_jets],
        reconstructed_bottom_mass[has_two_bottom_jets]
    )

    # Reconstructed leptons, with the electrons and muons merged
    reconstructed_lepton_pt, reconstructed_lepton_eta, reconstructed_lepton_phi = merge_leptons(
        electron_pt,
        electron_eta,
        electron_phi,
        muon_pt,
        muon_eta,
        muon_phi
    )

    has_two_reconstructed_leptons = ak.num(reconstructed_lepton_pt) >= 2
    reconstructed_dilepton_mass = invariant_mass(
        reconstructed_lepton_pt[has_two_reconstructed_leptons],
        reconstructed_lepton_eta[has_two_reconstructed_leptons],
        reconstructed_lepton_phi[has_two_reconstructed_leptons],
        ak.zeros_like(reconstructed_lepton_pt[has_two_reconstructed_leptons])
    )

    # Line shape fits
    # The parton fit checks that the sample carries the width the generator was run with
    parton_parameters, parton_errors = fit_z_width(parton_dilepton_mass)
    fitted_width = abs(parton_parameters[2])

    reconstructed_parameters, reconstructed_errors, voigt = fit_z_resolution(
        reconstructed_dilepton_mass,
        Z_WIDTH_GENERATOR
    )

    # Delta R matching with each method
    # The bottom quarks are matched to every jet rather than to the b-tagged ones:
    # the Delphes b-tagging already assigns the flavour by looking for a b parton
    # within delta R 0.5 of the jet axis, so matching to tagged jets alone would
    # fold the tagging efficiency into the angular resolution
    n_parton_bottoms = int(ak.sum(ak.num(parton_bottom_pt)))
    n_parton_leptons = int(ak.sum(ak.num(parton_lepton_pt)))

    n_jets = int(ak.sum(ak.num(jet_pt)))
    n_bottom_jets = int(ak.sum(ak.num(reconstructed_bottom_pt)))

    bottom_results = {}
    lepton_results = {}

    for method in MATCHING_METHODS:
        distances, btags, without_candidate, duplicate_events, matchable_events = match_to_reconstructed(
            parton_bottom_eta,
            parton_bottom_phi,
            jet_eta,
            jet_phi,
            method,
            reconstructed_flag = jet_btag
        )

        within = distances < BOTTOM_MATCHING_DELTA_R

        bottom_results[method] = {
            "distances": distances,
            "without_candidate": without_candidate,
            "duplicate_events": duplicate_events,
            "matchable_events": matchable_events,
            "matched": int(np.sum(within)),
            "matched_and_tagged": int(np.sum(within & (btags == 1)))
        }

        distances, _, without_candidate, duplicate_events, matchable_events = match_to_reconstructed(
            parton_lepton_eta,
            parton_lepton_phi,
            reconstructed_lepton_eta,
            reconstructed_lepton_phi,
            method
        )

        lepton_results[method] = {
            "distances": distances,
            "without_candidate": without_candidate,
            "duplicate_events": duplicate_events,
            "matchable_events": matchable_events,
            "matched": int(np.sum(distances < LEPTON_MATCHING_DELTA_R))
        }

    dilepton_minimum = 60
    dilepton_maximum = 120

    bbbar_minimum = 0
    bbbar_maximum = 250

    delta_r_minimum = 0
    delta_r_maximum = 5

    print()
    print("Object counts")
    print(f"{'bottom quarks from a Higgs':<28}: {n_parton_bottoms:>9}")
    print(f"{'leptons from a Z':<28}: {n_parton_leptons:>9}")
    print(f"{'reconstructed jets':<28}: {n_jets:>9}")
    print(f"{'b-tagged among them':<28}: {n_bottom_jets:>9}")

    print()
    print("Dilepton invariant mass")
    print(f"{'parton mean':<28}: {parton_dilepton_mass.mean():>9.2f} GeV")
    print(f"{'parton standard deviation':<28}: {parton_dilepton_mass.std():>9.2f} GeV")
    print(f"{'reconstructed mean':<28}: {reconstructed_dilepton_mass.mean():>9.2f} GeV")
    print(f"{'reconstructed std dev':<28}: {reconstructed_dilepton_mass.std():>9.2f} GeV")

    print()
    print(f"Breit-Wigner fit to the parton-level Z, {Z_FIT_MINIMUM}-{Z_FIT_MAXIMUM} GeV")
    print(f"{'fitted mass':<28}: {parton_parameters[1]:>9.3f} +/- {parton_errors[1]:.3f} GeV")
    print(f"{'fitted width':<28}: {fitted_width:>9.3f} +/- {parton_errors[2]:.3f} GeV")
    print(f"{'generator width':<28}: {Z_WIDTH_GENERATOR:>9.3f} GeV")
    print(f"{'PDG mass':<28}: {Z_MASS_PDG:>9.3f} GeV")
    print(f"{'PDG width':<28}: {Z_WIDTH_PDG:>9.3f} GeV")

    print()
    print("Voigt fit to the reconstructed Z")
    print(f"{'natural width fixed to':<28}: {Z_WIDTH_GENERATOR:>9.3f} GeV")
    print(f"{'fitted mass':<28}: {reconstructed_parameters[1]:>9.3f} +/- {reconstructed_errors[1]:.3f} GeV")
    print(f"{'detector resolution':<28}: {abs(reconstructed_parameters[2]):>9.3f} +/- {reconstructed_errors[2]:.3f} GeV")

    print()
    print("bbbar invariant mass")
    print(f"{'parton mean':<28}: {parton_bbbar_mass.mean():>9.2f} GeV")
    print(f"{'parton standard deviation':<28}: {parton_bbbar_mass.std():>9.2f} GeV")
    print(f"{'reconstructed mean':<28}: {reconstructed_bbbar_mass.mean():>9.2f} GeV")
    print(f"{'reconstructed std dev':<28}: {reconstructed_bbbar_mass.std():>9.2f} GeV")

    print()
    print(f"Bottom partons matched to every jet, criterion delta R < {BOTTOM_MATCHING_DELTA_R}")
    print(f"{'partons':<28}: {n_parton_bottoms:>9}")
    print(f"{'without a candidate':<28}: {bottom_results['nearest']['without_candidate']:>9}")
    print(
        f"{'duplicate assignments':<28}: "
        f"{bottom_results['nearest']['duplicate_events']:>9}"
        f" of {bottom_results['nearest']['matchable_events']} events"
    )
    print(f"{'':<28}{'entries':>10}{'median':>9}{'matched':>10}{'of partons':>12}{'tagged':>9}{'tag rate':>10}")
    for method in MATCHING_METHODS:
        result = bottom_results[method]
        tag_rate = 100*result["matched_and_tagged"]/result["matched"] if result["matched"] else 0

        print(
            f"{method:<28}"
            f"{len(result['distances']):>10}"
            f"{np.median(result['distances']):>9.3f}"
            f"{result['matched']:>10}"
            f"{100*result['matched']/n_parton_bottoms:>11.1f}%"
            f"{result['matched_and_tagged']:>9}"
            f"{tag_rate:>9.1f}%"
        )

    print()
    print(f"Lepton partons matched, criterion delta R < {LEPTON_MATCHING_DELTA_R}")
    print(f"{'partons':<28}: {n_parton_leptons:>9}")
    print(f"{'without a candidate':<28}: {lepton_results['nearest']['without_candidate']:>9}")
    print(
        f"{'duplicate assignments':<28}: "
        f"{lepton_results['nearest']['duplicate_events']:>9}"
        f" of {lepton_results['nearest']['matchable_events']} events"
    )
    print(f"{'':<28}{'entries':>10}{'median':>9}{'matched':>10}{'of partons':>12}")
    for method in MATCHING_METHODS:
        result = lepton_results[method]

        print(
            f"{method:<28}"
            f"{len(result['distances']):>10}"
            f"{np.median(result['distances']):>9.3f}"
            f"{result['matched']:>10}"
            f"{100*result['matched']/n_parton_leptons:>11.1f}%"
        )

    print()
    print("Entries outside plot range")
    print(f"{'parton dilepton':<28}: {count_outside(parton_dilepton_mass, dilepton_minimum, dilepton_maximum):>6}/{len(parton_dilepton_mass):<7}")
    print(f"{'reconstructed dilepton':<28}: {count_outside(reconstructed_dilepton_mass, dilepton_minimum, dilepton_maximum):>6}/{len(reconstructed_dilepton_mass):<7}")
    print()
    print(f"{'parton bbbar':<28}: {count_outside(parton_bbbar_mass, bbbar_minimum, bbbar_maximum):>6}/{len(parton_bbbar_mass):<7}")
    print(f"{'reconstructed bbbar':<28}: {count_outside(reconstructed_bbbar_mass, bbbar_minimum, bbbar_maximum):>6}/{len(reconstructed_bbbar_mass):<7}")
    print()
    for method in MATCHING_METHODS:
        label = f"bottom {method}"
        print(
            f"{label:<28}: "
            f"{count_outside(bottom_results[method]['distances'], delta_r_minimum, delta_r_maximum):>6}"
            f"/{len(bottom_results[method]['distances']):<7}"
        )
    print()
    for method in MATCHING_METHODS:
        label = f"lepton {method}"
        print(
            f"{label:<28}: "
            f"{count_outside(lepton_results[method]['distances'], delta_r_minimum, delta_r_maximum):>6}"
            f"/{len(lepton_results[method]['distances']):<7}"
        )

    draw_histogram_with_fit(
        parton_dilepton_mass,
        breit_wigner,
        parton_parameters,
        40,
        Z_FIT_MINIMUM,
        Z_FIT_MAXIMUM,
        r"$M(\ell^+\ell^-)$ [GeV]",
        "Number of parton pairs",
        rf"Breit-Wigner, $\Gamma$ = {fitted_width:.3f} GeV",
        "dilepton_mass_parton_raw_count_fit.png"
    )

    draw_histogram_with_fit(
        reconstructed_dilepton_mass,
        voigt,
        reconstructed_parameters,
        40,
        Z_FIT_MINIMUM,
        Z_FIT_MAXIMUM,
        r"$M(\ell^+\ell^-)$ [GeV]",
        "Number of reconstructed pairs",
        rf"Voigt, $\sigma$ = {abs(reconstructed_parameters[2]):.3f} GeV",
        "dilepton_mass_reconstructed_raw_count_fit.png"
    )

    draw_histogram_pair(
        parton_dilepton_mass,
        reconstructed_dilepton_mass,
        "Parton level",
        "Reconstructed",
        50,
        dilepton_minimum,
        dilepton_maximum,
        r"$M(\ell^+\ell^-)$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "dilepton_mass_parton_vs_reconstructed_normalized_linear.png",
        density = True
    )

    draw_histogram_pair(
        parton_dilepton_mass,
        reconstructed_dilepton_mass,
        "Parton level",
        "Reconstructed",
        50,
        dilepton_minimum,
        dilepton_maximum,
        r"$M(\ell^+\ell^-)$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "dilepton_mass_parton_vs_reconstructed_normalized_log.png",
        density = True,
        log_scale = True
    )

    draw_histogram_pair(
        parton_bbbar_mass,
        reconstructed_bbbar_mass,
        "Parton level",
        "Reconstructed",
        50,
        bbbar_minimum,
        bbbar_maximum,
        r"$M(b\bar{b})$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "bbbar_mass_parton_vs_reconstructed_normalized.png",
        density = True
    )

    for method in MATCHING_METHODS:
        draw_histogram(
            bottom_results[method]["distances"],
            50,
            delta_r_minimum,
            delta_r_maximum,
            r"$\Delta R$(parton $b$, reconstructed jet)",
            "Number of pairs",
            f"bottom_{method}_delta_r_raw_count_linear.png"
        )

        draw_histogram(
            bottom_results[method]["distances"],
            50,
            delta_r_minimum,
            delta_r_maximum,
            r"$\Delta R$(parton $b$, reconstructed jet)",
            "Number of pairs",
            f"bottom_{method}_delta_r_raw_count_log.png",
            log_scale = True
        )

        draw_histogram(
            lepton_results[method]["distances"],
            50,
            delta_r_minimum,
            delta_r_maximum,
            r"$\Delta R$(parton $\ell$, reconstructed $\ell$)",
            "Number of pairs",
            f"lepton_{method}_delta_r_raw_count_linear.png"
        )

        draw_histogram(
            lepton_results[method]["distances"],
            50,
            delta_r_minimum,
            delta_r_maximum,
            r"$\Delta R$(parton $\ell$, reconstructed $\ell$)",
            "Number of pairs",
            f"lepton_{method}_delta_r_raw_count_log.png",
            log_scale = True
        )


if __name__ == "__main__":
    parton_reconstructed_comparison()
