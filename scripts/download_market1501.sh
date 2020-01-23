#!/bin/bash

fileid="1p1Jl-U4OTgmiVLUCAdyNjAo3sI56mtHa"
filename="market1501.tgz"
cd datasets
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

echo Extracting files...
tar -xzf market1501.tgz

rm market1501.tgz
rm cookie
cd -
