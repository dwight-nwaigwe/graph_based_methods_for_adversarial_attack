#!/bin/bash



oarsub -l /core=4,walltime=2:30:00 --project pr-deepneuro ./cluster_command1.sh -n "mapec_test_mob_svhn_aa_edge"
oarsub -l /core=4,walltime=2:30:00 --project pr-deepneuro ./cluster_command2.sh -n "mapec_test_mob_svhn_aa_modcc"
oarsub -l /core=4,walltime=2:30:00 --project pr-deepneuro ./cluster_command3.sh -n "mapec_test_mob_svhn_aa_outin"





