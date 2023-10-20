#!/bin/bash


var1="cluster_command"

for i in {0..9..1}
	do

		var2="$var1$i.sh"
 		#oarsub -l /core=8/gpu=1,walltime=10:00:00 --project pr-deepneuro ./$var2 -n "create_data_mobilenet$i"
 		oarsub -l /nodes=1,walltime=5 --project pr-deepneuro ./$var2      -n "create_data_mobilenet$i"
   	
   	done




