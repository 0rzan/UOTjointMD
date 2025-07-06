from utils import UtilsSparse
import numpy as np
from typing import List
from wasserstein import NMRSpectrum

def load_data():
    """
    Load NMR spectral data from files.

    Returns:
        Tuple[List[NMRSpectrum], NMRSpectrum]: Component spectra and mixture spectrum
    """
    components_names = ["Pinene", "Benzyl benzoate"]

    protons_list = [16, 12]

    dirname = "data"
    filename = dirname + "/" + "preprocessed_mix.csv"
    mix = np.loadtxt(filename, delimiter=",")

    how_many_components = len(components_names)
    names = ["comp" + str(i) for i in range(how_many_components)]

    files_with_components = [dirname + "/" + "preprocessed_comp0.csv", 
                             dirname + "/" + "preprocessed_comp1.csv"]
    spectra = []
    for i in range(how_many_components):
        filename = files_with_components[i]
        spectra.append(np.loadtxt(filename, delimiter=","))

    spectra2: List[NMRSpectrum] = []
    names = []
    for i in range(len(spectra)):
        spectra2.append(
            NMRSpectrum(
                confs=list(zip(spectra[i][:, 0], spectra[i][:, 1])),
                protons=protons_list[i],
            )
        )
        names.append("comp" + str(i))

    spectra = spectra2
    del spectra2
    mix = NMRSpectrum(confs=list(zip(mix[:, 0], mix[:, 1])))
    mix.trim_negative_intensities()
    mix.normalize()
    for sp in spectra:
        sp.trim_negative_intensities()
        sp.normalize()

    return spectra, mix


def main():
    spectra, mix = load_data()
    # Sample params
    N = 2100
    C = 20
    reg = 1.5
    regm1 = 230
    regm2 = 115
    eta_G = 1e-3
    eta_p = 1e-3
    tol = 1e-5
    gamma = 0.99
    max_iter = 1000

    sparse = UtilsSparse(spectra, mix, N, C, reg, regm1, regm2)
    G, p = sparse.joint_md(eta_G, eta_p, max_iter, tol=tol, gamma=gamma)
    print("Final p: ", p)
    print("True proportions: ", [0.3865, 1 - 0.3865])


if __name__ == "__main__":
    main()