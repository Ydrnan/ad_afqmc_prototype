from pyscf import gto, scf, cc
from ad_afqmc_prototype.wrapper.ucisd import Ucisd

mol = gto.M(atom="""
    N 2.5 0.0 0.0
    N 0.0 0.0 0.0
    """,
    basis="6-31g",
    verbose=3,
)
mf = scf.UHF(mol)
mf.kernel()

mo1 = mf.stability()[0]
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

mycc = cc.UCCSD(mf)
mycc.kernel()

afqmc = Ucisd(mycc)
mean, err, block_e_all, block_w_all = afqmc.kernel()
