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

# GiNaCのインクルードディレクトリとライブラリを探す
find_path(GINAC_INCLUDE_DIR ginac/ginac.h)
find_library(GINAC_LIBRARY ginac)

if(NOT GINAC_INCLUDE_DIR OR NOT GINAC_LIBRARY)
    message(FATAL_ERROR "GiNaC library not found")
endif()

# インクルードディレクトリとリンクディレクトリを追加
include_directories(${GINAC_INCLUDE_DIR})
link_directories(${GINAC_LIBRARY_DIR})

# 出力ディレクトリを親ディレクトリに設定
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# pybind11 モジュールを追加
pybind11_add_module(gate models/gate.cpp)
pybind11_add_module(bit models/bit.cpp)

# ターゲットにライブラリをリンク
target_link_libraries(gate PRIVATE Python3::Python ${CLN_LIBRARIES} ${GINAC_LIBRARY})
target_link_libraries(bit PRIVATE Python3::Python ${CLN_LIBRARIES} ${GINAC_LIBRARY})

# 通常のC++実行ファイルを追加
add_executable(my_calculation main.cpp)
target_link_libraries(my_calculation PRIVATE ${CLN_LIBRARIES} ${GINAC_LIBRARY})
