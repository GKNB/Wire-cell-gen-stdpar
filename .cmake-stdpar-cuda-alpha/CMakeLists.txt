cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(wire-cell-gen-stdpar CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS Off)

include_directories(SYSTEM $ENV{BOOST_INC})

set(BOOST_ROOT $ENV{BOOST_DIR} )
set(BOOST_INCLUDEDIR $ENV{BOOST_INC})
set(BOOST_LIBRARYDIR $ENV{BOOST_LIB})
set(Boost_USE_MULTITHREADED ON)

find_package(Boost REQUIRED COMPONENTS)

file(GLOB all_files ${PROJECT_SOURCE_DIR}/../src/*.cxx)

string(APPEND CMAKE_CXX_FLAGS "-g -stdpar -O3 -gpu=cuda12.0 -pedantic -Wall -cudalib=cublas,curand,cufft -DARCH_CUDA -DSTDPAR_ENABLE_CUDA")
string(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

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
