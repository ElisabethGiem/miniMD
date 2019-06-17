cmake_minimum_required(VERSION 3.12)
project(minimd)

add_executable(minimd)
target_compile_features(minimd PRIVATE cxx_std_14)

# Options
option(MINIMD_SINGLE_PRECISION, "Use single precision" OFF)
if (MINIMD_SINGLE_PRECISION)
  target_compile_definitions(minimd PRIVATE PRECISION=1)
else()
  target_compile_definitions(minimd PRIVATE PRECISION=2)
endif()

option(MINIMD_RESILIENCE "Use resilience" OFF)

target_compile_definitions(minimd PRIVATE PREC_TIMER)

# Dependencies
find_package(MPI REQUIRED)
target_link_libraries(minimd PRIVATE MPI::MPI_CXX)

find_package(Kokkos REQUIRED NO_CMAKE_PACKAGE_REGISTRY)

# Optional resilience
if(MINIMD_RESILIENCE)
  find_package(resilience REQUIRED)
  target_link_libraries(minimd PRIVATE Kokkos::resilience)
  target_compile_definitions(minimd PRIVATE MINIMD_RESILIENCE)
endif()


TARGET_LINK_KOKKOS(minimd PRIVATE)

target_link_libraries(minimd PRIVATE "-L${resilience_LINK_DIRECTORIES}")

# VeloC config
add_custom_command(TARGET minimd PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/minimd.cfg ${CMAKE_CURRENT_BINARY_DIR}/minimd.cfg)

# Other inputs

add_custom_command(TARGET minimd PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/in.lj.miniMD ${CMAKE_CURRENT_BINARY_DIR}/in.lj.miniMD)

add_custom_command(TARGET minimd PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/kokkos/Cu_u6.eam ${CMAKE_CURRENT_BINARY_DIR}/Cu_u6.eam)

add_subdirectory(kokkos)


get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()

