!scheme 
!1.0 
geomtyp=xyz
noorient
bohr
geometry={ 
4
Title
N 0.0 0.0 0.217070660699
H 0.0 1.770600776007 -0.506498208297
H -1.533385251982 -0.885300388003 -0.506498208297
H 1.533385251982 -0.885300388003 -0.506498208297
}
Set,spin = 0
Set,charge = 0
      basis=cc-pvdz-f12
{hf,so-sci} 
{ccsd(t)-f12c,scale_trip=1}
forces

exe_energy=energy
