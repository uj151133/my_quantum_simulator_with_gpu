cmake_minimum_required(VERSION 3.5)
cmake_policy(SET CMP0167 NEW)

project(Qmdd_Sim VERSION 0.1.0)

# C++20を使用する
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python 3.10を探す
find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# Eigenの検索とインクルード設定
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
# SIMD最適化のためのマクロを有効化
add_definitions(-DEIGEN_VECTORIZE)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    message(STATUS "x86_64 detected: Enabling AVX, FMA, and F16C optimizations.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx -mfma -mf16c")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*")
    message(STATUS "ARM architecture detected: Enabling NEON optimizations.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
else()
    message(WARNING "Unknown architecture, SIMD optimizations are not enabled.")
endif()

# 環境変数の設定をCMakeの変数に変換
set(YAML_CPP_INCLUDE_DIR /opt/homebrew/Cellar/yaml-cpp/0.8.0/include)
set(YAML_CPP_LIB_DIR /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib)

set(LIBOMP_LIB_DIR /opt/homebrew/opt/libomp/lib)
set(LIBOMP_INCLUDE_DIR /opt/homebrew/opt/libomp/include)

set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")

find_package(yaml-cpp REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS fiber context)

# GNUライブラリのインクルードとリンク
find_package(GSL REQUIRED)
find_library(GMP_LIB gmp)

# SIMD最適化フラグの設定
# if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
# elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm.*")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
# elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rv64gcv")
# elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "powerpc")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maltivec")
# endif()

# OpenMPの設定
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# インクルードディレクトリとリンクディレクトリを追加
include_directories(${YAML_CPP_INCLUDE_DIR} ${LIBOMP_INCLUDE_DIR})
link_directories(${YAML_CPP_LIB_DIR} ${LIBOMP_LIB_DIR})

# 出力ディレクトリを親ディレクトリに設定
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs/__pycache__)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

include_directories(src)
file(GLOB SOURCES "src/models/*.cpp" "src/common/*.cpp" "src/test/*/*.cpp" )

# 通常のC++実行ファイルを追加
add_executable(qmdd_sim main.cpp ${SOURCES})
target_include_directories(qmdd_sim PRIVATE ${GSL_INCLUDE_DIRS})

# 実行ファイルにリンクするライブラリを指定
target_link_libraries(qmdd_sim PRIVATE 
    yaml-cpp 
    Python3::Python 
    OpenMP::OpenMP_CXX 
    Boost::fiber Boost::context 
    GSL::gsl 
    ${GMP_LIB} 
    Eigen3::Eigen
)

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