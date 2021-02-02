#!/usr/bin/bash


# main entry point for webserver
# webserver handles everything before that, this here gets called with the inputs as .fasta file


signalp_6 = /tools/src/signalp-6.0

net_gpi=/tools/src/netgpi-1.0/

source /tools/opt/anaconda3/bin/activate $signalp_6/signalp_6_env

cd $signalp_6/src/signalp6/

python predict.py $@