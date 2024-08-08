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

# YAMLライブラリを追加
find_package(YAML-CPP REQUIRED)

# インクルードディレクトリとリンクディレクトリを追加
include_directories(${GINAC_INCLUDE_DIR} ${YAML_CPP_INCLUDE_DIRS})
link_directories(${GINAC_LIBRARY_DIR} ${YAML_CPP_LIBRARY_DIRS})

# 出力ディレクトリを親ディレクトリに設定
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs/__pycache__)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# modelsディレクトリをincludeディレクトリとして設定
include_directories(src)

# 通常のC++実行ファイルを追加
add_executable(qmdd_sim main.cpp src/models/gate.cpp src/models/bit.cpp src/models/qmdd.cpp src/models/uniqueTable.cpp src/common/mathUtils.cpp)
target_link_libraries(qmdd_sim PRIVATE ${CLN_LIBRARIES} ${GINAC_LIBRARY} ${YAML_CPP_LIBRARIES} Python3::Python)

# make run ターゲットの追加
add_custom_target(run
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
    DEPENDS qmdd_sim
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Running the application..."
)

# make debug ターゲットの追加
add_custom_target(debug
    COMMAND lldb --file ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
    DEPENDS qmdd_sim
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Debugging the application with lldb..."
)
