import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from params import hbar, B0, m_star, AU_TO_MEV, AU_TO_NM
from hamiltonian import build_H

def compute_spectrum(ky_array, Ve, nmax, nlevel=12):
    """
    ky_array  : 1D array of ky grid
    Ve        : electric potential
    nmax      : number of basis ## = 51 for trial
    nlevel    : number of displayed band
    """
    evals = []
    for ky in ky_array:
        H = build_H(ky*AU_TO_NM, Ve, nmax)
        evals_all, _ = eigh(H)
        evals.append(evals_all[:nlevel])

    return np.array(evals) * AU_TO_MEV      ## shape: (len(ky_array), nlevel)

ky_array = np.linspace(-0.5,0.5,101)        ## tunable: ky grid
ne = 0.0                                    ## tunable: electric field magnitude
nmax=201                                    ## tunable: number of basis
evals = compute_spectrum(ky_array, Ve=ne*(hbar*B0/m_star), nmax=nmax)
print("Computing spectrum ", "| ky grid:", evals.shape[0], "| basis number:", nmax)


## confirm evals built correctly
#if __name__ == '__main__':
#    print("Done. Shape:", evals.shape)
#    print("Eigenvalues at ky=-0.15(meV):", evals[7].round(3))

fig, ax = plt.subplots(figsize=(6, 5))
for band in range(evals.shape[1]):
    ## evals.shape[0]: len(ky_array);
    ## evals.shape[1]: nlevel
    ax.plot(ky_array, evals[:, band], color='steelblue', lw=1.2)
    ## evals[:, band]: every ky, certain (band) level

ax.set_xlabel('$k_y$(nm$^-1$)')
ax.set_ylabel('Energy(meV)')
ax.set_title('Fig. 2(b)')
ax.set_ylim(-5, 10)
ax.axvline(0, color='gray', lw=0.5, ls='dashed')
ax.axhline(0, color='gray', lw=0.5, ls='solid')
plt.tight_layout()
plt.savefig('spectrum_Ve0.png', dpi=150)
plt.show()
