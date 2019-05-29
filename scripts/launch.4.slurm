#!/bin/bash
#SBATCH -n 4
#SBATCH -c 48
#SBATCH --time=02:00:00
#SBATCH -o ./benchmark_minimd.out
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 4 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 48 -i ../kokkos/in.eam.miniMD 

