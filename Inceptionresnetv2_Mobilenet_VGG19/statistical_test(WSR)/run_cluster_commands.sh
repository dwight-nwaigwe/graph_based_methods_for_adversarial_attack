#!/bin/bash


var1="cluster_command"

for i in {0..9..1}
	do

		var2="$var1$i.sh"
 		oarsub -l /gpu=1,walltime=5:00:00 --project pr-deepneuro ./$var2  -n "WSR_make_data_autoattack$i"
   	
   	done




