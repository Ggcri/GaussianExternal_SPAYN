!scheme 
!1.0 
geomtyp=xyz
noorient
bohr
geometry={ 
5
Title
C 0.0 0.0 0.0
H 1.165146497358 1.163256771233 1.165146497358
H -1.165146497358 -1.165146497358 1.165146497358
H -1.165146497358 1.165146497358 -1.165146497358
H 1.165146497358 -1.165146497358 -1.165146497358
}
Set,spin = 0
Set,charge = 0
      basis=cc-pvdz-f12

cDFTc=0.4210
cXHF=0.69
c2ab=0.5922
c2ss=0.0636
{DF-ks,pbex,p86,gridthr=1d-6,gridthr=3d-7;dh,cXHF,1.00-cDFTc}
EKS=ENERGY;
DF-mp2,ksfock,scsfacs=c2ab/(1.00-cDFTc),scsfact=c2ss/(1.00-cDFTc)
PT2=(EMP2_SCS - ENERGR)*(1.00-cDFTc)
!For normal MP2-F12, we need not to multiply SCSMP2DZ with 0.5790
exe_energy=EKS+PT2
