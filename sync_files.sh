## export PYSCF_PATH=$HOME/Research/tmp_pyscf

export PYSCF_PATH=$1
cp pyscf/dft/*.py $PYSCF_PATH/pyscf/dft/
cp pyscf/mp/*.py $PYSCF_PATH/pyscf/mp/
cp pyscf/pbc/mp/*.py $PYSCF_PATH/pyscf/pbc/mp/

