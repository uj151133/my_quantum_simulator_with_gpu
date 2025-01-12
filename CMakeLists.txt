cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0167 NEW)

project(Qmdd_Sim VERSION 0.1.0)

project(Qmdd_Sim VERSION 0.1.0)

set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")
set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")


# C++20を使用する
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Python 3.10を探す
find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${Python3_INCLUDE_DIRS})
link_directories(${Python3_LIBRARY_DIRS})

# 環境変数の設定をCMakeの変数に変換
set(YAML_CPP_INCLUDE_DIR /opt/homebrew/Cellar/yaml-cpp/0.8.0/include)
set(YAML_CPP_LIB_DIR /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib)

find_package(yaml-cpp REQUIRED)
find_package(Boost 1.87.0 REQUIRED COMPONENTS 
    system
    thread
    fiber
    context
    filesystem
    program_options
)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "Boost not found")
endif()
find_package(Threads REQUIRED)

# GNUライブラリのインクルードとリンク
find_package(GSL REQUIRED)
if(APPLE)
    # MacOS (Homebrew) settings
    find_library(GMP_LIB
        NAMES gmp
        PATHS /opt/homebrew/lib
        REQUIRED
    )
    include_directories(/opt/homebrew/include)
elseif(UNIX AND NOT APPLE)
    # Ubuntu settings
    find_library(GMP_LIB
        NAMES gmp
        PATHS
        /usr/lib
        /usr/local/lib
        /usr/lib/x86_64-linux-gnu  # Ubuntu特有のパスを追加
        REQUIRED
    )
    find_path(GMP_INCLUDE_DIRS
        NAMES gmp.h
        PATHS
        /usr/include
        /usr/local/include
        REQUIRED
    )
    if(GMP_LIB AND GMP_INCLUDE_DIRS)
        include_directories(${GMP_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "GMP library not found. Please install libgmp-dev")
    endif()
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(SYSTEM 
    ${Boost_INCLUDE_DIRS}
    ${BOOST_ROOT}/include
    ${Python3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    /opt/homebrew/include
    ${CMAKE_SOURCE_DIR}/src
)

link_directories(
    ${YAML_CPP_LIB_DIR} 
    /opt/homebrew/lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${BOOST_ROOT}/include")

# 出力ディレクトリを親ディレクトリに設定
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs/__pycache__)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

include_directories(src)

file(GLOB SOURCES "src/models/*.cpp" "src/common/*.cpp" "src/test/*/*.cpp")


# 通常のC++実行ファイルを追加
add_executable(qmdd_sim main.cpp ${SOURCES})
target_include_directories(qmdd_sim PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(qmdd_sim PRIVATE 
    pthread 
    yaml-cpp::yaml-cpp 
    Python3::Python 
    ${Boost_LIBRARIES}
    GSL::gsl 
    ${GMP_LIB} 
    Threads::Threads
)

# make run ターゲットの追加
add_custom_target(run
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
    DEPENDS qmdd_sim
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Running the application ..."
)

# make debug ターゲットの追加
add_custom_target(debug
    COMMAND lldb --file ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
    DEPENDS qmdd_sim
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Debugging the application with lldb ..."
)