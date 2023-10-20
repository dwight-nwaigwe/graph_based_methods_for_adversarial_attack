#!/bin/bash


var1="cluster_command"

for i in {0..9..1}
	do

		var2="$var1$i.sh"
 		oarsub -l /core=2,walltime=7:00:00 --project pr-deepneuro ./$var2 -n "create_data_model1_projected$i"
   	
   	done




