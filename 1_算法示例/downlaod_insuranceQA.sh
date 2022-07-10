#!/bin/bash

wget -L https://github.com/shuzi/insuranceQA/archive/refs/heads/master.zip
unzip master.zip
mkdir datas
mv insuranceQA-master/V2 datas/
cd ./datas/V2
gunzip *.gz
cd ..
cd ..
rm -rf insuranceQA-master/
rm -f master.zip
cp ./datas/V2/vocabulary ./datas/
