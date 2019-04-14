#!/bin/bash
# https://drive.google.com/file/d/16zZJ5f5qOJcgg-cPfmAdso8al-MSWiwu/view?usp=sharing
fileid="16zZJ5f5qOJcgg-cPfmAdso8al-MSWiwu"
filename="market1501.tgz"
cd ./datasets/
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
