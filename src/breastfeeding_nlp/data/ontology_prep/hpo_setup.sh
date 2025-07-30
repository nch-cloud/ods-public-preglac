#!/bin/sh
DIR=$1

# curl https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo > $DIR/hp.obo
python3.10 $DIR/hpo_syns.py $DIR/FPO.txt > $DIR/tmp
python3.10 $DIR/standardize_syn_map.py $DIR/tmp > $DIR/hpo_synonyms.txt
rm $DIR/tmp
python3.10 $DIR/hpo_names.py $DIR/FPO.txt > $DIR/hpo_term_names.txt 

