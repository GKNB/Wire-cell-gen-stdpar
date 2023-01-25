cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(wire-cell-gen-stdpar CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)
find_package(Boost REQUIRED COMPONENTS)
set(Boost_USE_MULTITHREADED ON)

file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)

set_source_files_properties(/home/twang/myWork/wire-cell-gen-stdpar/src/BinnedDiffusion_transform.cxx PROPERTIES COMPILE_FLAGS -stdpar)
set_source_files_properties(/home/twang/myWork/wire-cell-gen-stdpar/src/ImpactTransform.cxx PROPERTIES COMPILE_FLAGS -stdpar)
set_source_files_properties(/home/twang/myWork/wire-cell-gen-stdpar/src/AtomicWrapper.cxx PROPERTIES COMPILE_FLAGS -stdpar)
set_source_files_properties(/home/twang/myWork/wire-cell-gen-stdpar/src/StdparArray.cxx PROPERTIES COMPILE_FLAGS -stdpar)

#find_package(CUDA REQUIRED)

string(APPEND CMAKE_CXX_FLAGS "-g -O3 -pedantic -Wall -cudalib=cublas,curand,cufft -DARCH_CUDA -DSTDPAR_ENABLE_CUDA")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined -stdpar")

add_library(WireCellGenStdpar SHARED ${all_files})
target_include_directories(WireCellGenStdpar
  PRIVATE
    ${PROJECT_SOURCE_DIR}/../inc
    $ENV{OMPRNG}
    $ENV{EIGEN_INC}
    $ENV{JSONCPP_INC}
    $ENV{JSONNET_INC}
    $ENV{SPDLOG_INC}
    $ENV{WIRECELL_INC}
)
set_target_properties(WireCellGenStdpar
	PROPERTIES COMPILE_OPTIONS "-DEIGEN_NO_CUDA;-DEIGEN_DONT_VECTORIZE")

target_link_directories(WireCellGenStdpar PRIVATE  $ENV{JSONCPP_LIB} $ENV{WIRECELL_LIB})
target_link_libraries(WireCellGenStdpar PRIVATE  jsoncpp WireCellIface WireCellUtil)

#add_subdirectory(test)
