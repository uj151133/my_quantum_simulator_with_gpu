cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0057 NEW)

project(Qmdd_Sim VERSION 0.1.0)

# C++20を使用する
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python 3.10を探す
find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# pybind11のパスを手動で設定
set(pybind11_DIR ${CMAKE_PREFIX_PATH}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 REQUIRED)

# CLNライブラリのパスを設定
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
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs/__pycache__)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# modelsディレクトリをincludeディレクトリとして設定
include_directories(src)

# pybind11モジュールを追加
pybind11_add_module(gate src/models/gate.cpp)
pybind11_add_module(bit src/models/bit.cpp)
pybind11_add_module(calculation src/common/calculation.cpp)
pybind11_add_module(qmdd src/models/qmdd.cpp)
pybind11_add_module(uniqueTable src/models/uniqueTable.cpp)
pybind11_add_module(mathUtils src/common/mathUtils.cpp)

# ターゲットにライブラリをリンク
target_link_libraries(gate PRIVATE Python3::Python ${CLN_LIBRARIES} ${GINAC_LIBRARY})
target_link_libraries(bit PRIVATE Python3::Python ${CLN_LIBRARIES} ${GINAC_LIBRARY})
target_link_libraries(calculation PRIVATE Python3::Python ${CLN_LIBRARIES} ${GINAC_LIBRARY})
target_link_libraries(qmdd PRIVATE Python3::Python ${CLN_LIBRARIES} ${GINAC_LIBRARY})
target_link_libraries(uniqueTable PRIVATE Python3::Python ${CLN_LIBRARIES} ${GINAC_LIBRARY})
target_link_libraries(mathUtils PRIVATE Python3::Python ${CLN_LIBRARIES} ${GINAC_LIBRARY})

# 通常のC++実行ファイルを追加
add_executable(qmdd_sim main.cpp src/models/gate.cpp src/models/bit.cpp src/models/qmdd.cpp src/models/uniqueTable.cpp src/common/calculation.cpp src/common/mathUtils.cpp)
target_link_libraries(qmdd_sim PRIVATE ${CLN_LIBRARIES} ${GINAC_LIBRARY} pybind11::module Python3::Python)

# make run ターゲットの追加
add_custom_target(run
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Running the application..."
)

# make debug ターゲットの追加
add_custom_target(debug
    COMMAND lldb --file ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Debugging the application with lldb..."
)