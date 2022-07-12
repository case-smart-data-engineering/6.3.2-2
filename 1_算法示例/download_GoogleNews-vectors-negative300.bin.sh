#!/bin/bash

#wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
#gzip -d GoogleNews-vectors-negative300.bin.gz

wget -c "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/13774556/GoogleNewsvectorsnegative300.bin"
mv GoogleNewsvectorsnegative300.bin GoogleNews-vectors-negative300.bin
