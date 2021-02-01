#!/bin/bash
config0="test"
outdir="outputs"
noiseExp=-4
seed=1934
oldconfig=$config0
while [[ $(perl -e "print $noiseExp <= -2") == "1" ]]
do
	noise=$(perl -e "print 10**$noiseExp")
	newconfig=$config0'_1e'$noiseExp
	sed -i "s/\[$oldconfig\]/\[$newconfig\]/g" microstructure_gpr.input
	sed -i "s/kernel_noise = .*/kernel_noise = $noise/g" microstructure_gpr.input
	sed -i "s/seed = .*/seed = $seed/g" microstructure_gpr.input
	python microstructure_gpr.py $newconfig &> $outdir"/"$newconfig".out"
	echo "noise=1.e$noiseExp"
	tail -n 13 $outdir"/"$newconfig".out"
	#python microstructure_gpr.py $newconfig
	oldconfig=$newconfig
	noiseExp=$(perl -e "printf '%.3f',$noiseExp+(1.0/5.0)")
	#((seed++))
done
