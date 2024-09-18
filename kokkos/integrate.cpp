/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */
//#define PRINTDEBUG(a) a
#define PRINTDEBUG(a)
#include "stdio.h"
#include "integrate.h"
#include "math.h"
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <fstream>

#ifdef KOKKOS_ENABLE_RESILIENT_EXECUTION
#include <resilience/Resilience.hpp>
#include <resilience/openMP/ResHostSpace.hpp>
#include <resilience/openMP/ResOpenMP.hpp>
#include <resilience/openMP/OpenMPResSubscriber.hpp>
#endif

#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
   #include <mpi.h>
   #ifdef KOKKOS_ENABLE_HDF5
      #define CHECKPOINT_FILESPACE KokkosResilience::HDF5Space
      #ifdef KOKKOS_ENABLE_HDF5_PARALLEL
         bool serial_io = false;
      #else
         bool serial_io = true;
      #endif
   #else
      bool serial_io = true;
      #define CHECKPOINT_FILESPACE KokkosResilience::StdFileSpace
   #endif
#endif

Integrate::Integrate() {sort_every=20;}
Integrate::~Integrate() {}

void Integrate::setup()
{
  dtforce = 0.5 * dt;
}

void Integrate::initialIntegrate(int step)
{
#ifdef KOKKOS_ENABLE_RESILIENT_EXECUTION
  Kokkos::parallel_for(Kokkos::RangePolicy< KokkosResilience::ResOpenMP, TagInitialIntegrate>(0,nlocal), *this);
#else      
  Kokkos::parallel_for(Kokkos::RangePolicy<TagInitialIntegrate>(0,nlocal), *this);
#endif
}

KOKKOS_INLINE_FUNCTION
void Integrate::operator() (TagInitialIntegrate, const int& i) const {
  v(i,0) += dtforce * f(i,0);
  v(i,1) += dtforce * f(i,1);
  v(i,2) += dtforce * f(i,2);
  x(i,0) += dt * v(i,0);
  x(i,1) += dt * v(i,1);
  x(i,2) += dt * v(i,2);
}

void Integrate::finalIntegrate(int step)
{
  Kokkos::parallel_for(Kokkos::RangePolicy<TagFinalIntegrate>(0,nlocal), *this);
}

KOKKOS_INLINE_FUNCTION
void Integrate::operator() (TagFinalIntegrate, const int& i) const {
  v(i,0) += dtforce * f(i,0);
  v(i,1) += dtforce * f(i,1);
  v(i,2) += dtforce * f(i,2);
}

void Integrate::run(Atom &atom, Force* force, Neighbor &neighbor,
                    Comm &comm, Thermo &thermo, Timer &timer, 
                    const int restart_, std::string root_path)
{
  int i, n;
  
  //TODO RESILIENT TEST COUNTER: TO BE REOMVED
  int integrate_counter=0;

  /*
#ifdef KOKKOS_ENABLE_RESILIENT_EXECUTION
  std::ofstream res_minimd_posfile;
  res_minimd_posfile.open("/home/eagiem/resilient-miniMD/miniMD/test/res_minimd_posfile.txt");
#else
  std::ofstream minimd_posfile;
  minimd_posfile.open("/scratch/minimd_posfile.txt");
#endif
*/
  
  comm.timer = &timer;
  timer.array[TIME_TEST] = 0.0;

  int check_safeexchange = comm.check_safeexchange;

  mass = atom.mass;
  dtforce = dtforce / mass;

    int next_sort = sort_every>0?sort_every:ntimes+1;
    int nStart = 0;

#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
    CHECKPOINT_FILESPACE sfs;
    if (comm.me == 0) printf("manual checkpoint using cp mirror: %s \n", sfs.name());
    auto x_cp = Kokkos::create_chkpt_mirror( sfs, atom.x );
    auto v_cp = Kokkos::create_chkpt_mirror( sfs, atom.v );
    auto f_cp = Kokkos::create_chkpt_mirror( sfs, atom.f );
    nStart = restart_;

// Load from restart ...
    if (nStart > 0) {
         std::string cp_path = root_path;
         cp_path+=(std::string)"data";
         if ( comm.nprocs > 1 &&
              !std::is_same<CHECKPOINT_FILESPACE, KokkosResilience::StdFileSpace>::value &&
              !serial_io )
            KokkosResilience::DirectoryManager<CHECKPOINT_FILESPACE>::
                   set_checkpoint_directory(comm.me == 0 ? true : false, cp_path.c_str(), (int)((nStart / 10) * 10));
         else
            KokkosResilience::DirectoryManager<CHECKPOINT_FILESPACE>::
                   set_checkpoint_directory( true , cp_path.c_str(), (int)((nStart / 10) * 10), comm.me);
         // need to resize the views to match the checkpoint files ... 
         CHECKPOINT_FILESPACE::restore_all_views();
    }
#endif
#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
    resilience_context->register_alias( "velocity", "atom::v_copy" );
    resilience_context->register_alias( "atom", "atom::x_copy" );

//#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
    nStart = KokkosResilience::latest_version( *resilience_context, "initial_integrate" );
    if ( nStart < 0 )
        nStart = 0;
#endif

    for(int n = nStart; n < ntimes; n++) {
      Kokkos::fence();

//TODO: ignore for now
#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
      int nlocal = atom.nlocal;
      int nmax = atom.nmax;
      KokkosResilience::checkpoint( *resilience_context, "initial_integrate", n, [&, KR_CHECKPOINT( nlocal ), KR_CHECKPOINT( nmax ),
                                                                                  _tmp_atom=atom,
                                                                                  _tmp_neigh=neighbor,
                                                                                  _tmp_comm=comm,
                                                                                  KR_CHECKPOINT_THIS]() mutable {
//#endif
      atom.nlocal = nlocal;
      atom.nmax = nmax;
#endif
      x = atom.x;
      v = atom.v;
      f = atom.f;
      xold = atom.xold;
      nlocal = atom.nlocal;

//TODO: res + timer      

  for ( int i = 0; i < x.extent(0); i++){

  /*
#ifdef KOKKOS_ENABLE_RESILIENT_EXECUTION
  res_minimd_posfile << integrate_counter + 1 << " x(" <<i<<"):\n";
  res_minimd_posfile << x(i, 0) << "\n" << x(i,1) << "\n" << x(i, 2) << "\n"; 
#else
  minimd_posfile << integrate_counter + 1 << " x(" <<i<< "):\n";   
  minimd_posfile << x(i, 0) << "\n" << x(i,1) << "\n" << x(i, 2) << "\n";

#endif
*/
  }

      std::cout << x(1,0) << "," << x(1,1) << "," << x(1,2) << "\n";
      const auto start{std::chrono::steady_clock::now()};
      initialIntegrate(n);
      integrate_counter++;
      const auto stop{std::chrono::steady_clock::now()};
      const auto time = stop-start;
      std::cout << "Initial integrate loop " << integrate_counter << " took " << time.count() << " nanoseconds.\n\n";

      timer.stamp();

      if((n + 1) % neighbor.every) {

	//TODO: res + timer      
        comm.communicate(atom);
        timer.stamp(TIME_COMM);

      } else {
          if(check_safeexchange) {
              double d_max = 0;

              for(i = 0; i < atom.nlocal; i++) {
                double dx = (x(i,0) - xold(i,0));

                if(dx > atom.box.xprd) dx -= atom.box.xprd;

                if(dx < -atom.box.xprd) dx += atom.box.xprd;

                double dy = (x(i,1) - xold(i,1));

                if(dy > atom.box.yprd) dy -= atom.box.yprd;

                if(dy < -atom.box.yprd) dy += atom.box.yprd;

                double dz = (x(i,2) - xold(i,2));

                if(dz > atom.box.zprd) dz -= atom.box.zprd;

                if(dz < -atom.box.zprd) dz += atom.box.zprd;

                double d = dx * dx + dy * dy + dz * dz;

                if(d > d_max) d_max = d;
              }

              d_max = sqrt(d_max);

              if((d_max > atom.box.xhi - atom.box.xlo) || (d_max > atom.box.yhi - atom.box.ylo) || (d_max > atom.box.zhi - atom.box.zlo))
                printf("Warning: Atoms move further than your subdomain size, which will eventually cause lost atoms.\n"
                "Increase reneighboring frequency or choose a different processor grid\n"
                "Maximum move distance: %lf; Subdomain dimensions: %lf %lf %lf\n",
                d_max, atom.box.xhi - atom.box.xlo, atom.box.yhi - atom.box.ylo, atom.box.zhi - atom.box.zlo);

          }

          timer.stamp_extra_start();
          //TODO: res + timer
	  comm.exchange(atom);
          if(n+1>=next_sort) {
            atom.sort(neighbor);
            next_sort +=  sort_every;
          }
	  //TODO: res + timer
          comm.borders(atom);
          timer.stamp_extra_stop(TIME_TEST);
          timer.stamp(TIME_COMM);

        Kokkos::fence();

	Kokkos::Profiling::pushRegion("neighbor::build");
        //TODO: res + timer
	neighbor.build(atom);
	Kokkos::Profiling::popRegion();

        timer.stamp(TIME_NEIGH);
      }

      Kokkos::Profiling::pushRegion("force");
      force->evflag = (n + 1) % thermo.nstat == 0;
      //TODO: res + timer
      force->compute(atom, neighbor, comm, comm.me);
      Kokkos::Profiling::popRegion();

      timer.stamp(TIME_FORCE);

      if(neighbor.halfneigh && neighbor.ghost_newton) {
        //TODO: res + timer
	comm.reverse_communicate(atom);

        timer.stamp(TIME_COMM);
      }

      v = atom.v;
      f = atom.f;
      nlocal = atom.nlocal;

      Kokkos::fence();

      //TODO: res + timer
      finalIntegrate(n);

      //TODO: res + timer
      if(thermo.nstat) thermo.compute(n + 1, atom, neighbor, force, timer, comm);
//TODO: ignore here to end of checkpoint for now
#ifdef KOKKOS_ENABLE_MANUAL_CHECKPOINT
      if ( n % 10 == 0 ) {
         std::string cp_path = root_path;
         cp_path+=(std::string)"data";
         Kokkos::fence();
         if ( comm.nprocs > 1 && 
              !std::is_same<CHECKPOINT_FILESPACE, KokkosResilience::StdFileSpace>::value &&
              !serial_io ) 
            KokkosResilience::DirectoryManager<CHECKPOINT_FILESPACE>::
                      set_checkpoint_directory(comm.me == 0 ? true : false, cp_path.c_str(), n);
         else
            KokkosResilience::DirectoryManager<CHECKPOINT_FILESPACE>::
                      set_checkpoint_directory( true , cp_path.c_str(), n, comm.me);
         CHECKPOINT_FILESPACE::checkpoint_views();
         MPI_Barrier( MPI_COMM_WORLD );
         //if (comm.me == 0) printf("checkpoint complete: %d \n", n); 
      } else {
         //if (comm.me == 0) printf("compute only iteration: %d \n", n); 
      }
#endif
#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
      if ( ( fail_iter > 0 ) && ( n == fail_iter ) && is_fail_node ) {
        printf("Intentionally killing rank on iteration %d.\n", n );
        MPI_Abort( MPI_COMM_WORLD, 400 );
      }
//#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
      } );
#endif
    }
/*
#ifdef KOKKOS_ENABLE_RESILIENT_EXECUTION
  res_minimd_posfile.close();
#else
  minimd_posfile.close();
#endif
  */
}
