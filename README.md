# PT2 Variants

Variants of second-order perturbation theory (PT2) for computing the electronic correlation energy.
Note that all methods are called some flavor of "MP2" in the code, but they can be used
with DFT initial Hamiltonians as well (GL2).

Methods include:

* $\kappa$-MP2: Single-parameter regularization by Lee *et al.* (J. Chem. Theory Comput. 2018, 14, 5203)
* $\lambda$-MP2: Density functional-based regularization (experimental)
* AC-MP2: Adiabatic connection functionals using the size-consistent OSMI method (to be published)
* RMP2: Restricted open-shell MP2 by Knowles *et al.* (Chem. Phys. Lett. 1991, 186, 130). Note that in the code it is called ROMP2, not to be mistaken with the ROMP theory of Amos *et al.* (Chem. Phys. Lett. 1991, 185, 256).

Also included is the `pyscf.acmp.ueg` module for performing PT2 calculations on the
(non-spin-polarized) the uniform electron gas.

All methods under `pyscf.acmp.pbc` are experimental and not thoroughly tested.
