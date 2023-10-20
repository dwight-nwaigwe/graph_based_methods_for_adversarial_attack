#!/bin/bash



oarsub -l /core=4,walltime=1:00:00 --project pr-deepneuro ./cluster_command1.sh -n "model1_projected_edges"
oarsub -l /core=4,walltime=1:00:00 --project pr-deepneuro ./cluster_command2.sh -n "mod1_projected_outin"
oarsub -l /core=4,walltime=1:00:00 --project pr-deepneuro ./cluster_command3.sh -n "mod1_projected_modcc"





