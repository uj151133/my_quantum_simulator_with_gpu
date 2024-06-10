cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0057 NEW)

project(My_Calculation VERSION 0.1.0)

# Python 3.10 を探す
find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# pybind11 のパスを手動で設定
set(pybind11_DIR ${CMAKE_PREFIX_PATH}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)

# CLN ライブラリのパスを設定
set(CLN_DIR /usr/local/include/cln/)
list(APPEND CMAKE_PREFIX_PATH ${CLN_DIR})
find_package(CLN REQUIRED)

# GiNaCのパスを探す
set(GiNaC_DIR /usr/local/include/ginac/)
set(GiNaC_INCLUDE_DIRS /usr/local/include/ginac/)
set(GiNaC_LIBRARY_DIRS /usr/local/lib/)

list(APPEND CMAKE_PREFIX_PATH ${GiNaC_DIR})
message(STATUS "GiNaC_DIR: ${GiNaC_DIR}")
find_package(GiNaC REQUIRED CONFIG)

include_directories(${GiNaC_INCLUDE_DIRS})
link_directories(${GiNaC_LIBRARY_DIRS})
include_directories(${CLN_INCLUDE_DIRS})

# 出力ディレクトリを親ディレクトリに設定
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# pybind11 モジュールを追加
pybind11_add_module(gate models/gate.cpp)
pybind11_add_module(bit models/bit.cpp)

message(STATUS "GiNaC_INCLUDE_DIRS: ${GiNaC_INCLUDE_DIRS}")
message(STATUS "GiNaC_LIBRARIES: ${GiNaC_LIBRARIES}")
message(STATUS "GiNaC_LIBRARY_DIRS: ${GiNaC_LIBRARY_DIRS}")

# ターゲットにライブラリをリンク
target_link_libraries(gate PRIVATE Python3::Python ${CLN_LIBRARIES} ${GiNaC_LIBRARIES})
target_link_libraries(bit PRIVATE Python3::Python ${CLN_LIBRARIES} ${GiNaC_LIBRARIES})

# 通常のC++実行ファイルを追加
add_executable(my_calculation main.cpp)
target_link_libraries(my_calculation PRIVATE ${CLN_LIBRARIES} ${GiNaC_LIBRARIES})
