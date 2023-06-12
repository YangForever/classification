#   This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk
#
#   NOTE hash dollar is a scheduler directive not a comment.


# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=2G
#$ -l h_rt=3600 
#$ -l gpu=true


#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N MyTESTJOBNAME

#The code you want to run now goes here.

hostname
date 

source /share/apps/source_files/python/python-3.9.5.source
python3 classification/train.py

date
