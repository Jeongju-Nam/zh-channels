from common import (
    TTBAR_FILE,
    ZH_FILE,
    count_outside,
    draw_histogram_pair,
    extract_masses
)


def compare_zh_ttbar():
    zh_total, zh_dilepton, zh_bbbar = extract_masses(ZH_FILE)
    ttbar_total, ttbar_dilepton, ttbar_bbbar = extract_masses(TTBAR_FILE)

    dilepton_minimum = 0
    dilepton_maximum = 300

    bbbar_minimum = 0
    bbbar_maximum = 500

    zh_label = r"$ZH \to \ell\ell b\bar{b}$"
    ttbar_label = r"$t\bar{t} \to b\ell\nu b\ell\nu$"

    print()
    print("Event counts")
    print(f"{'ZH total events':<28}: {zh_total:>9}")
    print(f"{'ZH with dilepton pair':<28}: {len(zh_dilepton):>9}")
    print(f"{'ZH with two b-jets':<28}: {len(zh_bbbar):>9}")
    print(f"{'ttbar total events':<28}: {ttbar_total:>9}")
    print(f"{'ttbar with dilepton pair':<28}: {len(ttbar_dilepton):>9}")
    print(f"{'ttbar with two b-jets':<28}: {len(ttbar_bbbar):>9}")

    print()
    print("Dilepton invariant mass")
    print(f"{'ZH mean':<28}: {zh_dilepton.mean():>9.2f} GeV")
    print(f"{'ZH standard deviation':<28}: {zh_dilepton.std():>9.2f} GeV")
    print(f"{'ttbar mean':<28}: {ttbar_dilepton.mean():>9.2f} GeV")
    print(f"{'ttbar standard deviation':<28}: {ttbar_dilepton.std():>9.2f} GeV")

    print()
    print("bbbar invariant mass")
    print(f"{'ZH mean':<28}: {zh_bbbar.mean():>9.2f} GeV")
    print(f"{'ZH standard deviation':<28}: {zh_bbbar.std():>9.2f} GeV")
    print(f"{'ttbar mean':<28}: {ttbar_bbbar.mean():>9.2f} GeV")
    print(f"{'ttbar standard deviation':<28}: {ttbar_bbbar.std():>9.2f} GeV")

    print()
    print("Entries outside plot range")
    print(f"{'ZH dilepton':<28}: {count_outside(zh_dilepton, dilepton_minimum, dilepton_maximum):>6}/{len(zh_dilepton):<7}")
    print(f"{'ttbar dilepton':<28}: {count_outside(ttbar_dilepton, dilepton_minimum, dilepton_maximum):>6}/{len(ttbar_dilepton):<7}")
    print(f"{'ZH bbbar':<28}: {count_outside(zh_bbbar, bbbar_minimum, bbbar_maximum):>6}/{len(zh_bbbar):<7}")
    print(f"{'ttbar bbbar':<28}: {count_outside(ttbar_bbbar, bbbar_minimum, bbbar_maximum):>6}/{len(ttbar_bbbar):<7}")

    draw_histogram_pair(
        zh_dilepton,
        ttbar_dilepton,
        zh_label,
        ttbar_label,
        50,
        dilepton_minimum,
        dilepton_maximum,
        r"$M(\ell^+\ell^-)$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "dilepton_mass_normalized_linear.png",
        density = True
    )

    draw_histogram_pair(
        zh_dilepton,
        ttbar_dilepton,
        zh_label,
        ttbar_label,
        50,
        dilepton_minimum,
        dilepton_maximum,
        r"$M(\ell^+\ell^-)$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "dilepton_mass_normalized_log.png",
        density = True,
        log_scale = True
    )

    draw_histogram_pair(
        zh_bbbar,
        ttbar_bbbar,
        zh_label,
        ttbar_label,
        50,
        bbbar_minimum,
        bbbar_maximum,
        r"$M(b\bar{b})$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "bbbar_mass_normalized_linear.png",
        density = True
    )

    draw_histogram_pair(
        zh_bbbar,
        ttbar_bbbar,
        zh_label,
        ttbar_label,
        50,
        bbbar_minimum,
        bbbar_maximum,
        r"$M(b\bar{b})$ [GeV]",
        r"Probability density [GeV$^{-1}$]",
        "bbbar_mass_normalized_log.png",
        density = True,
        log_scale = True
    )


if __name__ == "__main__":
    compare_zh_ttbar()
