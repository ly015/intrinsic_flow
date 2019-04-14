#!/bin/bash

fileid="1LbibHhhF7xA7G3hHoHj9I-MvCByzdkvr"
filename="deepfashion.tgz"
cd datasets
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

echo Extracting files...
tar -xzf deepfashion.tgz

rm deepfashion.tgz
rm cookie
cd -
