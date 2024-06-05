cmake_minimum_required(VERSION 3.5)
# CMP0057 ポリシーを設定
cmake_policy(SET CMP0057 NEW)

# プロジェクト名とバージョン
project(My_Calculation VERSION 0.1.0)

# Python 3 を探す
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Python include directory
execute_process(
    COMMAND python -c "from sysconfig import get_paths as gp; print(gp()['include'])"
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Python library
execute_process(
    COMMAND python -c "from sysconfig import get_paths as gp; print(gp()['platlib'] + '/libpython3.10.dylib')"
    OUTPUT_VARIABLE PYTHON_LIBRARY
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Export variables
set(PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR} CACHE PATH "Python include directory")
set(PYTHON_LIBRARY ${PYTHON_LIBRARY} CACHE FILEPATH "Python library")

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARY})

# pybind11 のパスを手動で設定
set(pybind11_DIR ${CMAKE_PREFIX_PATH}/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)
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
link_directories(${GiNaC_LIBRARY_DIRS})
include_directories(${CLN_INCLUDE_DIRS})
link_directories(${CLN_LIBRARY_DIRS})

# pybind11 モジュールを追加
pybind11_add_module(gate models/gate.cpp)
pybind11_add_module(bit models/bit.cpp)

# ターゲットにライブラリをリンク
target_link_libraries(gate PRIVATE ${CLN_LIBRARIES} ${GiNaC_LIBRARIES})
target_link_libraries(bit PRIVATE ${CLN_LIBRARIES} ${GiNaC_LIBRARIES})

# EXAMPLE_VERSION_INFO は setup.py によって定義され、
# C++ コードに渡される (VERSION_INFO) ものです。
# target_compile_definitions(myadd PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
