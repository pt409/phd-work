#!/bin/bash
config0="test"
outdir="outputs"
noiseExp=-3
seed=1934
oldconfig=$config0
while [[ $(perl -e "print $noiseExp < -1") == "1" ]]
do
	noise=$(perl -e "print 10**$noiseExp")
	newconfig=$config0'_1e'$noiseExp
	sed -i "s/\[$oldconfig\]/\[$newconfig\]/g" microstructure_gpr.input
	sed -i "s/kernel_noise = .*/kernel_noise = $noise/g" microstructure_gpr.input
	sed -i "s/seed = .*/seed = $seed/g" microstructure_gpr.input
	python microstructure_gpr.py $newconfig &> $outdir"/"$newconfig".out"
	#python microstructure_gpr.py $newconfig
	oldconfig=$newconfig
	noiseExp=$(perl -e "printf '%.3f',$noiseExp+(1.0/3.0)")
	((seed++))
done
