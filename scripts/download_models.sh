#!/bin/bash

fileid="1QHcb1QBGVmGginpsYmer-Q5Aq0HNtBHv"
filename="pretrained_models.tgz"
cd checkpoints
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

echo Extracting files...
tar -xzf pretrained_models.tgz

rm pretrained_models.tgz
rm cookie
cd -
