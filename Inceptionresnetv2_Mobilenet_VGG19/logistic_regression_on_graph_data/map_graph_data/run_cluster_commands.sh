#!/bin/bash



oarsub -l /core=4,walltime=2:00:00 --project pr-deepneuro ./cluster_command1.sh -n "mob_autoattack_edges"
oarsub -l /core=4,walltime=2:00:00 --project pr-deepneuro ./cluster_command2.sh -n "mob_autoattack_outin"
oarsub -l /core=4,walltime=2:00:00 --project pr-deepneuro ./cluster_command3.sh -n "mob_autoattack_modcc"





