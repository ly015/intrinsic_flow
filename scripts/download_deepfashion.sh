#!/bin/bash

fileid="1_VJh4TlVRMoIgkZSaicgICfye3s5eaVJ"
filename="deepfashion.tgz"
cd datasets
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

echo Extracting files...
tar -xzf deepfashion.tgz

rm deepfashion.tgz
rm cookie
cd -
