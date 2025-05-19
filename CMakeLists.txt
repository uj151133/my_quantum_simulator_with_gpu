cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(Qmdd_Sim VERSION 0.1.0)

# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

if(APPLE)
    set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
    set(JVM_LIB_PATH "/opt/homebrew/Cellar/openjdk@21/21.0.7/libexec/openjdk.jdk/Contents/Home/lib/server/libjvm.dylib")

    set(BOOST_MIN_VERSION "1.87.0")
    set(YAML_CPP_INCLUDE_DIR /opt/homebrew/Cellar/yaml-cpp/0.8.0/include)
    set(YAML_CPP_LIB_DIR /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib)
elseif(UNIX AND NOT APPLE)
    # Ubuntu用のBoost設定
    set(BOOST_MIN_VERSION "1.83.0")
    set(JAVA_HOME "/usr/lib/jvm/java-21-openjdk-amd64")
    include_directories(${JAVA_HOME}/include)
    include_directories(${JAVA_HOME}/include/linux)
    find_library(JVM_LIB jvm HINTS ${JAVA_HOME}/lib/server REQUIRED)
    set(JVM_LIB_PATH ${JVM_LIB})
endif()

# C++20を使用する
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${Python3_INCLUDE_DIRS})

# 環境変数の設定をCMakeの変数に変換


# 環境変数の設定をCMakeの変数に変換

find_package(yaml-cpp REQUIRED)
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS 
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
find_package(TBB REQUIRED)

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
    /opt/homebrew/opt/openjdk/include
    /opt/homebrew/opt/openjdk/include/darwin
    ${CMAKE_SOURCE_DIR}/src
)

link_directories(
    ${Python3_LIBRARY_DIRS}
    ${YAML_CPP_LIB_DIR}
    /opt/homebrew/lib
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${BOOST_ROOT}/include")

# 出力ディレクトリを親ディレクトリに設定
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs/__pycache__)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

include_directories(src)

enable_language(C)
enable_language(ASM)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    message(STATUS "Building for x86_64 architecture")
    file(GLOB C_SRC "src/atomic/x86_64/atomic_x86_64.c")
    file(GLOB ASM_SRC "src/atomic/x86_64/atomic_x86_64.s")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i386")
    message(STATUS "Building for x86 (32-bit) architecture")
    file(GLOB C_SRC "src/atomic/x86/atomic_x86.c")
    file(GLOB ASM_SRC "src/atomic/x86/atomic_x86.s")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "Building for ARM64 architecture")
    file(GLOB C_SRC "src/atomic/arm64/atomic_arm64.c")
    file(GLOB ASM_SRC "src/atomic/arm64/atomic_arm64.s")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
# file(GLOB C_SRC "src/atomic/atomic.c")
file(GLOB SOURCES "src/models/*.cpp" "src/common/*.cpp" "src/modules/*.cpp" "src/test/*/*.cpp" )


# 通常のC++実行ファイルを追加
add_executable(qmdd_sim main.cpp ${SOURCES} ${C_SRC} ${ASM_SRC})
target_include_directories(qmdd_sim PRIVATE ${GSL_INCLUDE_DIRS})
target_link_libraries(qmdd_sim PRIVATE
    pthread
    yaml-cpp::yaml-cpp
    Python3::Python
    ${Boost_LIBRARIES}
    GSL::gsl
    ${GMP_LIB}
    Threads::Threads
    ${JVM_LIB_PATH}
    xxhash
    TBB::tbb
)

# make run ターゲットの追加
add_custom_target(run
    COMMAND DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/openjdk@21/21.0.7/libexec/openjdk.jdk/Contents/Home/lib/server:$ENV{DYLD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
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
