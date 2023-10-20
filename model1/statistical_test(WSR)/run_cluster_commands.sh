#!/bin/bash
#for model1 only need < 1 hr

var1="cluster_command"

for i in {0..9..1}
	do

		var2="$var1$i.sh"
 		oarsub -l /core=2/nodes=1,walltime=0:20:00 --project pr-deepneuro ./$var2 -n "model1_projected_$i"
   	
   	done




