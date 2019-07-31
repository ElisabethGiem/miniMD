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

#ifdef MINIMD_RESILIENCE
   #include <resilience/Resilience.hpp>
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

#ifdef KOKKOS_ENABLE_RESILIENT_EXECUTION
   #define DEVICE_EXECUTION_SPACE Kokkos::ResCuda
#else
   #ifdef KOKKOS_ENABLE_CUDA
      #define DEVICE_EXECUTION_SPACE Kokkos::Cuda
   #else
      #define DEVICE_EXECUTION_SPACE Kokkos::OpenMP
   #endif
#endif

Integrate::Integrate() {sort_every=20;}
Integrate::~Integrate() {}

void Integrate::setup()
{
  dtforce = 0.5 * dt;
}

void Integrate::initialIntegrate()
{
  Kokkos::parallel_for(Kokkos::RangePolicy<DEVICE_EXECUTION_SPACE, TagInitialIntegrate>(0,nlocal), *this);
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

void Integrate::finalIntegrate()
{
  Kokkos::parallel_for(Kokkos::RangePolicy<DEVICE_EXECUTION_SPACE, TagFinalIntegrate>(0,nlocal), *this);
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

    for(n = nStart; n < ntimes; n++) {
#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
    #ifdef KR_ENABLE_TRACING
      auto iter_time = KokkosResilience::Util::begin_trace< KokkosResilience::Util::IterTimingTrace< std::string > >( *resilience_context, "step", n );
    #endif
#endif

      Kokkos::fence();

      x = atom.x;
      v = atom.v;
      f = atom.f;
      xold = atom.xold;
      nlocal = atom.nlocal;

#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
      KokkosResilience::checkpoint( *resilience_context, "initial", n, [self = *this]() mutable {
        self.initialIntegrate();
      }, KokkosResilience::filter::nth_iteration_filter< 10 >{} );
#else
      initialIntegrate();
#endif

      timer.stamp();

      if((n + 1) % neighbor.every) {

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
          comm.exchange(atom);
          if(n+1>=next_sort) {
            atom.sort(neighbor);
            next_sort +=  sort_every;
          }
          comm.borders(atom);
          timer.stamp_extra_stop(TIME_TEST);
          timer.stamp(TIME_COMM);

        Kokkos::fence();

	Kokkos::Profiling::pushRegion("neighbor::build");
        neighbor.build(atom);
	Kokkos::Profiling::popRegion();

        timer.stamp(TIME_NEIGH);
      }

      Kokkos::Profiling::pushRegion("force");
      force->evflag = (n + 1) % thermo.nstat == 0;
      force->compute(atom, neighbor, comm, comm.me);
      Kokkos::Profiling::popRegion();

      timer.stamp(TIME_FORCE);

      if(neighbor.halfneigh && neighbor.ghost_newton) {
        comm.reverse_communicate(atom);

        timer.stamp(TIME_COMM);
      }

      v = atom.v;
      f = atom.f;
      nlocal = atom.nlocal;

      Kokkos::fence();

#ifdef KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT
      KokkosResilience::checkpoint( *resilience_context, "final", n, [self = *this]() mutable {
        self.finalIntegrate();
      }, KokkosResilience::filter::nth_iteration_filter< 10 >{} );
#else
      finalIntegrate();
#endif

      if(thermo.nstat) thermo.compute(n + 1, atom, neighbor, force, timer, comm);
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
    }
}
