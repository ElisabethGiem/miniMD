cmake_minimum_required(VERSION 3.12)
project(minimd)

add_executable(minimd)
add_executable(minimd_resil)
    # Require C++14
if (CMAKE_CXX_COMPILER MATCHES ".*nvcc_wrapper" )
    message("nvcc wrapper requires that CXX standard by set in kokkos build")
else()
   target_compile_features(minimd PUBLIC cxx_std_14)
   target_compile_features(minimd_resil PUBLIC cxx_std_14)
endif()

# Options
option(MINIMD_SINGLE_PRECISION, "Use single precision" OFF)
if (MINIMD_SINGLE_PRECISION)
  target_compile_definitions(minimd PRIVATE PRECISION=1)
else()
  target_compile_definitions(minimd PRIVATE PRECISION=2)
  target_compile_definitions(minimd_resil PRIVATE PRECISION=2)
endif()

option(MINIMD_AUTOMATIC_CHECKPOINT "Use resilience automatic checkpointing" OFF)
option(MINIMD_MANUAL_CHECKPOINT "Use resilience manual checkpointing" OFF)
option(MINIMD_RESILIENT_EXECUTION "Use resilient execution space" ON)

target_compile_definitions(minimd PRIVATE PREC_TIMER)
target_compile_definitions(minimd_resil PRIVATE PREC_TIMER)


# Dependencies
if (MINIMD_USE_MPISTUB)
   find_library( mpi_stub_ NAMES libmpi_stubs.a libmpi_stubs mpi_stubs HINTS ${PROJECT_SOURCE_DIR}/kokkos/MPI-Stubs )
   #add_library(mpi_stubs)
   #target_link_libraries(minimd PRIVATE mpi_stubs)
   target_link_libraries(minimd PRIVATE ${mpi_stub_})
   target_include_directories(minimd PUBLIC
                           ${PROJECT_SOURCE_DIR}/kokkos/MPI-Stubs
                           )
   target_link_libraries(minimd_resil PRIVATE ${mpi_stub_})
   target_include_directories(minimd_resil PUBLIC
                           ${PROJECT_SOURCE_DIR}/kokkos/MPI-Stubs
			   )
else()
   find_package(MPI REQUIRED)
   target_link_libraries(minimd PRIVATE MPI::MPI_CXX)
   target_link_libraries(minimd_resil PRIVATE MPI::MPI_CXX)
endif()

find_package(Kokkos REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

# Optional resilience
if(MINIMD_AUTOMATIC_CHECKPOINT OR MINIMD_MANUAL_CHECKPOINT OR MINIMD_RESILIENT_EXECUTION)
  find_package(resilience REQUIRED)
  target_link_libraries(minimd_resil PRIVATE Kokkos::resilience)
  target_compile_definitions(minimd_resil PRIVATE MINIMD_RESILIENCE)
  if(MINIMD_AUTOMATIC_CHECKPOINT)
     target_compile_definitions(minimd_resil PRIVATE KOKKOS_ENABLE_AUTOMATIC_CHECKPOINT)
  endif()
  if(MINIMD_MANUAL_CHECKPOINT)
     target_compile_definitions(minimd_resil PRIVATE KOKKOS_ENABLE_MANUAL_CHECKPOINT)
  endif()
  if(MINIMD_RESILIENT_EXECUTION)
     target_compile_definitions(minimd_resil PRIVATE KOKKOS_ENABLE_RESILIENT_EXECUTION)
  endif()
endif()

target_link_libraries(minimd PRIVATE Kokkos::kokkos)
target_link_libraries(minimd_resil PRIVATE Kokkos::kokkos)

# VeloC config
add_custom_command(TARGET minimd PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/minimd.cfg ${CMAKE_CURRENT_BINARY_DIR}/minimd.cfg)
add_custom_command(TARGET minimd PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/minimd.json ${CMAKE_CURRENT_BINARY_DIR}/minimd.json)

add_custom_command(TARGET minimd_resil PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/minimd.cfg ${CMAKE_CURRENT_BINARY_DIR}/minimd.cfg)
add_custom_command(TARGET minimd_resil PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/minimd.json ${CMAKE_CURRENT_BINARY_DIR}/minimd.json)



# Other inputs

add_custom_command(TARGET minimd PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/in.lj.miniMD ${CMAKE_CURRENT_BINARY_DIR}/in.lj.miniMD)

add_custom_command(TARGET minimd PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/Cu_u6.eam ${CMAKE_CURRENT_BINARY_DIR}/Cu_u6.eam)

add_custom_command(TARGET minimd_resil PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/in.lj.miniMD ${CMAKE_CURRENT_BINARY_DIR}/in.lj.miniMD)

add_custom_command(TARGET minimd_resil PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/Cu_u6.eam ${CMAKE_CURRENT_BINARY_DIR}/Cu_u6.eam)


add_subdirectory(kokkos)

# Install rules
include(GNUInstallDirs)

install(TARGETS minimd_resil minimd)

