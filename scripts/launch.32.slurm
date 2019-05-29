#!/bin/bash
#SBATCH -n 32
#SBATCH -c 6
#SBATCH --time=02:00:00
#SBATCH -o ./benchmark_minimd.out
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 
rm -r ./data
mpirun -n 32 ./miniMD --half_neigh 0 -s 100 --ntypes 1 -t 6 -i ../kokkos/in.eam.miniMD 

