#!/usr/bin/env bash
# Code snippet from Github repository
# Repository: https://github.com/MTG/mtg-jamendo-dataset
# File: scripts/download/extract_all.sh
# Commit: 8cec497
cd $1
for i in *.tar
do
    tar -xvf $i && rm $i
done
cd -
