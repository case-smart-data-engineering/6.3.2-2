#!/bin/bash

wget -L https://github.com/shuzi/insuranceQA/archive/refs/heads/master.zip
unzip master.zip
mkdir datas
mv insuranceQA-master/V2 datas/
rm -rf insuranceQA-master/
rm -f master.zip
cp ./datas/V2/vocabulary ./datas/
