!scheme 
!1.0 
geometry={
C1
C2,C1,r2
C3,C1,r2,C2,a19
C4,C1,r4,C2,a5,C3,d2
C5,C2,r7,C3,a5,C4,d2
C6,C3,r4,C4,a5,C5,d2
C7,C4,r8,C5,a12,C6,d1
C8,C5,r9,C6,a10,C7,d2
C9,C6,r8,C7,a6,C8,d1
C10,C7,r11,C8,a5,C9,d2
C11,C8,r12,C9,a5,C10,d2
C12,C9,r11,C10,a5,C11,d2
C13,C10,r15,C11,a9,C12,d1
C14,C11,r16,C12,a13,C13,d2
C15,C12,r16,C13,a4,C14,d2
C16,C13,r20,C14,a8,C15,d1
C17,C14,r21,C15,a15,C16,d2
C18,C15,r13,C16,a11,C17,d2
C19,C16,r1,C17,a5,C18,d1
C20,C17,r5,C18,a1,C19,d1
C21,C18,r5,C19,a17,C20,d1
C22,C19,r5,C20,a5,C21,d1
C23,C20,r21,C21,a10,C22,d2
C24,C21,r19,C22,a8,C23,d2
H1,C22,r22,C23,a7,C24,d1
H2,C23,r6,C24,a20,H1,d2
H3,C24,r14,H1,a3,H2,d2
H4,H1,r24,H2,a10,H3,d1
H5,H2,r24,H3,a14,H4,d2
H6,H3,r10,H4,a14,H5,d2
H7,H4,r18,H5,a2,H6,d2
H8,H5,r17,H6,a16,H7,d1
H9,H6,r23,H7,a16,H8,d2
H10,H7,r3,H8,a18,H9,d2
H11,H8,r10,H9,a19,H10,d2
H12,H9,r3,H10,a16,H11,d2
}
r1=1.369688270239 angstrom
r2=1.424108969411 angstrom
r3=2.460824410174 angstrom
r4=2.466629090534 angstrom
r5=2.489240894971 angstrom
r6=2.712181134174 angstrom
r7=2.848217938821 angstrom
r8=3.758744316557 angstrom
r9=4.262701477949 angstrom
r10=4.774556463200 angstrom
r11=4.916586446772 angstrom
r12=5.677185017077 angstrom
r13=5.681179972608 angstrom
r14=5.828038365516 angstrom
r15=6.177534147454 angstrom
r16=6.484590581038 angstrom
r17=6.744429770476 angstrom
r18=6.760046210199 angstrom
r19=7.050868242847 angstrom
r20=7.350851464528 angstrom
r21=7.477370073057 angstrom
r22=8.135870202643 angstrom
r23=8.269774377868 angstrom
r24=9.549112927633 angstrom
a1=10.5549132256 degree
a2=15.0662562353 degree
a3=18.1865592649 degree
a4=22.1314703191 degree
a5=29.9999999962 degree
a6=40.8453680686 degree
a7=49.4201426129 degree
a8=49.4450867767 degree
a9=53.1719068372 degree
a10=59.9999999948 degree
a11=70.5549132223 degree
a12=70.8453680744 degree
a13=71.0657351628 degree
a14=75.0662562177 degree
a15=79.4450867818 degree
a16=89.9999999926 degree
a17=100.5549132255 degree
a18=105.0662562455 degree
a19=120.0000000000 degree
a20=173.5811458033 degree
d1=-180.0000000000 degree
d2=0.0000000000 degree
Set,spin = 0
Set,charge = 0
      basis=cc-pvdz
{rhf,so-sci}
{ccsd(t)-f12b,df_basis=avdz-f12/mp2fit,df_basis_exch=avdz-f12/jkfit,ri_basis=avdz/jkfit}
ccsd_energy = energy


basis=cc-pwcvtz

{rhf,so-sci}
{mp2}

mp2_fc = energy


{rhf,so-sci}
{mp2;core}
mp2_ae = energy


! Composite energy calculation according to scheme, without hf_energy
exe_energy = ccsd_energy + mp2_ae - mp2_fc



!normalmode
!symmetry=auto
!zmat
!restart
!reference_fc=error_dependent
!energy_error_grad=1e-9
!characteristic_length=0.1
!fakekey
! scf=xqc  


