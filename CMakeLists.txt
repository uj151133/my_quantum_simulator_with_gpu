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
set(CLN_DIR /opt/local/include/cln/)
list(APPEND CMAKE_PREFIX_PATH ${CLN_DIR})
find_package(CLN REQUIRED)

# GiNaCのパスを探す
set(GiNaC_DIR /opt/local/include/ginac/)
list(APPEND CMAKE_PREFIX_PATH ${GiNaC_DIR})
find_package(GiNaC REQUIRED CONFIG)

include_directories(${GiNaC_INCLUDE_DIRS})
include_directories(${CLN_INCLUDE_DIRS})

# pybind11 モジュールを追加
pybind11_add_module(gate models/gate.cpp)
pybind11_add_module(bit models/bit.cpp)

# ターゲットにライブラリをリンク
target_link_libraries(gate PRIVATE Python3::Python ${CLN_LIBRARIES} ${GiNaC_LIBRARIES})
target_link_libraries(bit PRIVATE Python3::Python ${CLN_LIBRARIES} ${GiNaC_LIBRARIES})
