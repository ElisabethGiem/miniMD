target_sources(minimd PRIVATE
               ${PROJECT_SOURCE_DIR}/kokkos/atom.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/comm.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/force_eam.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/force_lj.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/input.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/integrate.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/ljs.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/neighbor.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/output.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/setup.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/thermo.cpp
               ${PROJECT_SOURCE_DIR}/kokkos/timer.cpp
               )
