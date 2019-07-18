#!/bin/bash

TOT_TIME=0

cat ./benchmark_minimd.out | grep PERF_SUMMARY > benchmark_results.out
let loop_cnt=0
while read -r result; do

MD_TIME=`echo $result | awk '{print $5}'`
TOT_TIME=`echo "${TOT_TIME} + ${MD_TIME}" | bc -l`
echo $TOT_TIME, $MD_TIME, $result

let loop_cnt+=1
done < ./benchmark_results.out
AVG_MD_TIME=`echo "${TOT_TIME} / ${loop_cnt}" | bc -l`

echo "MiniMD: ${AVG_MD_TIME}"
