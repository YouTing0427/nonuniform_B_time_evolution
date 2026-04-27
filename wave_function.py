import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite import hermval
from math import factorial
from scipy.linalg import eigh
from params import hbar, B0, L, m_star, w_c, AU_TO_MEV, AU_TO_NM
from hamiltonian import build_H

nmax=21        ## number of basis

def probability_mat(ky, Ve, level):
    H = build_H(ky*AU_TO_NM, Ve, nmax=nmax)
    _, evecs = eigh(H)
    return evecs[:,level]

print(probability_mat(0.15, 0, [0,1,2]))


def QHO_basis(x, n):
    #norm = (2**n * factorial(n))**(-1/2) * (m_star*w_c / (np.pi*hbar))**(1/4)
    coeffs = [0]*n +[1]
    H = hermval(np.sqrt(m_star*w_c/hbar) * x, coeffs) * np.exp(-m_star*w_c * x**2 / (2*hbar))
    norm = np.linalg.norm(H)
    return 1/norm * H 


def probability(ky, Ve, level, x):
    coeffs = probability_mat(ky, Ve, level)
    p0 = np.zeros_like(x)

    for nbasis in range(nmax):
        p0 += coeffs[nbasis] * QHO_basis(x, nbasis)
    
    return p0**2


x = np.linspace(-L, L, 101)
ky = 0.15
Ve = 0
level = 0
fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x*AU_TO_NM, probability(ky, Ve, level, x), color='blue', lw=1.2)
ax.set_xlabel('x(nm)')
ax.set_ylabel('probability')
ax.set_title('Fig. 2(c)')
ax.axvline(0, color='gray', lw=0.5, ls='dashed')
ax.axhline(0, color='gray', lw=0.5, ls='solid')
plt.tight_layout()
plt.savefig('Probability_ky-0.15.png', dpi=150)
plt.show()