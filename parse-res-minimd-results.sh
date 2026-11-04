#!/bin/bash

OUT_DIR=random-injection-100-timesteps
OUT_FNAME=parsed-resminimd-random-100-timesteps.txt

declare -a nthreads=(1 2 4 8 16 28 56)
nruns=($(seq 1 10))

for t in "${nthreads[@]}"; do
  for i in "${nruns[@]}"; do

     # MiniMD, deterministic striped recovery
        
     FNAME=./"${OUT_DIR}"/resminimd_N1_n1_t"${t}"_nsteps100_i"${i}".txt
     if [ -f "${FNAME}" ]; then
       initial_integrate_time=$(grep "All initial integrate loops took" "${FNAME}" | awk '{print $6}')
       final_integrate_time=$(grep "All final integrate loops took" "${FNAME}" | awk '{print $6}')
       total_integrate_time=$(grep "Total parallel integration time was" "${FNAME}" | awk '{print $6}')
       total_errors=$(grep "The total number of errors inserted is" "${FNAME}" | awk '{print $8}')
       total_force=$(grep "Total force kernel time was" "${FNAME}" | awk '{print $6}')
       total_injection_time=$(grep "The value of ErrorInjectionTracking::total_error_time.count() is" "${FNAME}" | awk '{print $6}')
       total_time=$(grep -A 1 "perf/thread" "${FNAME}" | grep -v "perf/thread" | awk '{print $5}')
       echo "minimd,${i},${t},100,${total_errors},${initial_integrate_time},${final_integrate_time},${total_injection_time},${total_integrate_time},${total_force},${total_time}" >>"${OUT_DIR}"/"${OUT_FNAME}"
     else
       echo "${FNAME} not found"
     fi
  done
done

