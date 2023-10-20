#!/bin/bash



oarsub -l /core=4,walltime=1:00:00 --project pr-deepneuro ./cluster_command1.sh -n "model1_projected_edge"
oarsub -l /core=4,walltime=1:00:00 --project pr-deepneuro ./cluster_command2.sh -n "model1_projected_modcc"
oarsub -l /core=4,walltime=1:00:00 --project pr-deepneuro ./cluster_command3.sh -n "model1_projected_outin"





