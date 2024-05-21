cmake_minimum_required(VERSION 3.2)
# CMP0057 ポリシーを設定
cmake_policy(SET CMP0057 NEW)

# プロジェクト名とバージョン
project(myadd VERSION 0.1.0)

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

# pybind11 モジュールを追加
pybind11_add_module(myadd myadd.cpp)

# EXAMPLE_VERSION_INFO は setup.py によって定義され、
# C++ コードに渡される (VERSION_INFO) ものです。
# target_compile_definitions(myadd PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
