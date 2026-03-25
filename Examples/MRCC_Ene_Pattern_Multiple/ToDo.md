# New feature
For each computation for the MRCC implementation, if more than one energy pattern string is found in the preamble file, then, more parsing of the output file should be done and the energy contribution should be added and encapsulated in a unique variable with a summation operation. 
## EXAMPLE
For example in the mrcc_preamble.dat here you will find 
energy_pattern=Density-based correction [au]:
energy_pattern=CABS singles correction [au]:

So, in the first MINP output file, called MINP_1.out (also in the standard of the python implementation) you will need to find the strings 
 Density-based correction [au]:          -0.022050486502
 CABS singles correction [au]:           -0.004324385437

Take the two energy values and obtain in a unique energy variable the summation of the values. It is important that it is like this computation produces just one energy value, like one single computation is performed, and this energy value can be combined with the others. 

After that, the code will find in the preamble file the second section with the following :

!SECTION2
energy_pattern=KS energy + MP2 correction [au]:
basis=cc-pVDZ-F12

calc=scf
dft=user
5
0.3100 PBEx
0.4296 P86
0.6900 HFx
0.5785 MP2s
0.0799 MP2t

scfiguess=off

So the code shuold process as already does the MINP_2.out file and check for the string 
KS energy + MP2 correction [au]:
Particularly in the output this string will be found and the energy value to be extracted is reported here.
 KS energy + MP2 correction [au]:             -73.450191474046


The final energy value given in output should combine the first and this second variable following the !schema directives. 

In fact, at this point the code will have two energy variables to combine depending on the coefficients or the formula of the !schema block

# Testing

1. Run the script with testing mode on
2. Test the energy parsing, checking for the final summation value   

