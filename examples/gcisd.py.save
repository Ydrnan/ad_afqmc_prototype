from pyscf import gto, scf, cc
from ad_afqmc_prototype.wrapper.gcisd import Gcisd

mol = gto.M(atom="""
    N 0.0 0.0 0.0
    N 2.5 0.0 0.0
    """,
    basis="6-31g",
    verbose=3,
)
mf = scf.GHF(mol)
mf.kernel()

mo1 = mf.stability()
dm1 = mf.make_rdm1(mo1, mf.mo_occ)
mf = mf.run(dm1)
mf.stability()

mycc = cc.GCCSD(mf)
mycc.kernel()

afqmc = Gcisd(mycc)
mean, err, block_e_all, block_w_all = afqmc.kernel()
