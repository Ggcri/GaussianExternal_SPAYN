!scheme 
!1.0 
geomtyp=xyz
noorient
bohr
geometry={ 
4
Title
N 0.0 0.0 0.211394805237
H -0.0 1.779899001281 -0.493254545553
H -1.54143775128 -0.88994950064 -0.493254545553
H 1.54143775128 -0.88994950064 -0.493254545553
}
Set,spin = 0
Set,charge = 0
      basis=cc-pvdz-f12
{hf,so-sci} 
{ccsd(t)-f12c,scale_trip=1}
forces

exe_energy=energy
