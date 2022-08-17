#!/bin/bash

labeled_X="./labeled_DNA_methylation.csv" #WRITE FILENAME FOR LABELED DATASET
labeled_Y="./cancer_subtype.csv" #WRITE FILENAME FOR CANCER SUBTYPE
unlabeled_X="./unlabeled_DNA_methylation.csv" #WRITE FILENAME FOR UNLABELED DATASET

python3 preprocessing.py $labeled_X $labeled_Y $unlabeled_X