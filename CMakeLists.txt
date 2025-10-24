cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(Qmdd_Sim VERSION 0.1.0)

# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(GRAALVM_NATIVE_LIB_PATH "${CMAKE_SOURCE_DIR}/src/java/build/native/nativeCompile")
set(GRAALVM_NATIVE_LIB_SEARCH_NAME "operationcache")

if(APPLE)
    set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "15.0")
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isysroot ${CMAKE_OSX_SYSROOT}")

    set(GRAALVM_NATIVE_LIB_NAME "liboperationcache.dylib")

    set(BOOST_MIN_VERSION "1.87.0")
    set(YAML_CPP_INCLUDE_DIR /opt/homebrew/Cellar/yaml-cpp/0.8.0/include)
    set(YAML_CPP_LIB_DIR /opt/homebrew/Cellar/yaml-cpp/0.8.0/lib)
elseif(UNIX AND NOT APPLE)
    set(BOOST_MIN_VERSION "1.83.0")
    set(GRAALVM_NATIVE_LIB_NAME "liboperationcache.so")
elseif(WIN32)
    set(GRAALVM_NATIVE_LIB_NAME "operationcache.dll")
endif()

# C++20を使用する
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# インクルードディレクトリとリンクディレクトリの設定
include_directories(${Python3_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
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
find_package(antlr4-runtime REQUIRED)
# GNUライブラリのインクルードとリンク
find_package(GSL REQUIRED)

if(APPLE)
    # MacOS (Homebrew) settings
    find_path(ANTLR4_RUNTIME_INCLUDE_DIR antlr4-runtime.h
        HINTS
            /opt/homebrew/opt/antlr4-cpp-runtime/include
            /opt/homebrew/include
            /usr/local/include
        PATH_SUFFIXES
            antlr4-runtime
    )

    find_library(ANTLR4_RUNTIME_LIB NAMES antlr4-runtime
        HINTS
            /opt/homebrew/opt/antlr4-cpp-runtime/lib
            /opt/homebrew/lib
            /usr/local/lib
    )
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
    find_path(ANTLR4_RUNTIME_INCLUDE_DIR
        NAMES antlr4-runtime.h
        PATHS /usr/include /usr/local/include
        PATH_SUFFIXES antlr4-runtime
    )
    find_library(ANTLR4_RUNTIME_LIB
        NAMES antlr4-runtime
        PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
    )
    if(NOT ANTLR4_RUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR "antlr4-runtime headers not found. Install libantlr4-runtime-dev or set ANTLR4_RUNTIME_INCLUDE_DIR")
    endif()
    include_directories(${ANTLR4_RUNTIME_INCLUDE_DIR})
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(SYSTEM 
    ${Boost_INCLUDE_DIRS}
    ${BOOST_ROOT}/include
    ${Python3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    /opt/homebrew/include
    ${JAVA_HOME_ENV}/include
    ${JAVA_HOME_ENV}/include/darwin
    ${CMAKE_SOURCE_DIR}/src
    ${GRAALVM_NATIVE_LIB_PATH}
)

link_directories(
    ${Python3_LIBRARY_DIRS}
    ${YAML_CPP_LIB_DIR}
    /opt/homebrew/lib
    ${GRAALVM_NATIVE_LIB_PATH}
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

configure_file(${CMAKE_SOURCE_DIR}/config.yaml ${CMAKE_BINARY_DIR}/config.yaml COPYONLY)

find_library(GRAALVM_NATIVE_LIB
    NAMES ${GRAALVM_NATIVE_LIB_SEARCH_NAME}
    PATHS ${GRAALVM_NATIVE_LIB_PATH}
    NO_DEFAULT_PATH
)

if(NOT GRAALVM_NATIVE_LIB)
    message(WARNING "GraalVM Native Image library not found!")
    message(WARNING "Expected file: ${GRAALVM_NATIVE_LIB_PATH}/${GRAALVM_NATIVE_LIB_NAME}")
    message(WARNING "Please build the Java library first:")
    message(WARNING "  cd src/java")
    message(WARNING "  ./gradlew nativeCompile")
    
    # ファイルの存在確認
    if(EXISTS "${GRAALVM_NATIVE_LIB_PATH}/${GRAALVM_NATIVE_LIB_NAME}")
        message(STATUS "File exists but not found by find_library, using direct path")
        set(GRAALVM_NATIVE_LIB "${GRAALVM_NATIVE_LIB_PATH}/${GRAALVM_NATIVE_LIB_NAME}")
    endif()
endif()


file(GLOB SOURCES "src/models/*.cpp" "src/common/*.cpp" "src/modules/*.cpp" "src/test/*/*.cpp" "src/translator/*/*.cpp")


set(OPENQASM3_GEN_DIR ${CMAKE_SOURCE_DIR}/src/translator/OpenQASM3/gen)
file(GLOB OPENQASM3_GEN_SRC "${OPENQASM3_GEN_DIR}/*.cpp")



add_executable(qmdd_sim main.cpp ${SOURCES} ${C_SRC} ${ASM_SRC} ${OPENQASM3_GEN_SRC})
add_dependencies(qmdd_sim copy_config build_java_lib)
target_include_directories(qmdd_sim PRIVATE ${GSL_INCLUDE_DIRS} ${OPENQASM3_GEN_DIR} ${ANTLR4_RUNTIME_INCLUDE_DIR})
target_link_libraries(qmdd_sim PRIVATE
    pthread
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    Python3::Python
    ${Boost_LIBRARIES}
    GSL::gsl
    ${GMP_LIB}
    Threads::Threads
    xxhash
    TBB::tbb
    antlr4-runtime
    ${ANTLR4_RUNTIME_LIB}
    $<$<BOOL:${GRAALVM_NATIVE_LIB}>:${GRAALVM_NATIVE_LIB}>
)

if(APPLE)
    set(LIB_PATH_PREFIX "DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/openjdk@21/21.0.7/libexec/openjdk.jdk/Contents/Home/lib/server:\$DYLD_LIBRARY_PATH")
elseif(UNIX AND NOT APPLE)
    set(LIB_PATH_PREFIX "LD_LIBRARY_PATH=/home/ark/.sdkman/candidates/java/current/lib/server:/home/ark/.sdkman/candidates/java/current/lib:\$LD_LIBRARY_PATH")
else()
    set(LIB_PATH_PREFIX "")
endif()



# make run ターゲットの追加（通常モード）
if(APPLE)
    add_custom_target(run
        COMMAND DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/openjdk@21/21.0.7/libexec/openjdk.jdk/Contents/Home/lib/server:$ENV{DYLD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running the application in standalone mode..."
    )
elseif(UNIX AND NOT APPLE)
    add_custom_target(run
        COMMAND LD_LIBRARY_PATH=/home/ark/.sdkman/candidates/java/current/lib/server:/home/ark/.sdkman/candidates/java/current/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running the application in standalone mode..."
    )
else()
    add_custom_target(run
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running the application in standalone mode..."
    )
endif()

# make run-server ターゲットの追加（IPCサーバーモード）
if(APPLE)
    add_custom_target(run-server
        COMMAND DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/openjdk@21/21.0.7/libexec/openjdk.jdk/Contents/Home/lib/server:$ENV{DYLD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -s
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running the application in IPC server mode for GUI communication..."
    )
elseif(UNIX AND NOT APPLE)
    add_custom_target(run-server
        COMMAND LD_LIBRARY_PATH=/home/ark/.sdkman/candidates/java/current/lib/server:/home/ark/.sdkman/candidates/java/current/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -s
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running the application in IPC server mode for GUI communication..."
    )
else()
    add_custom_target(run-server
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -s
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running the application in IPC server mode for GUI communication..."
    )
endif()

# make debug ターゲットの追加
add_custom_target(debug
    COMMAND lldb --file ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
    DEPENDS qmdd_sim
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Debugging the application with lldb ..."
)

add_custom_target(copy_config
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/config.yaml
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config.yaml
    DEPENDS ${CMAKE_SOURCE_DIR}/config.yaml
    COMMENT "Copying config.yaml if it has changed"
)

add_custom_target(build_java_lib
    COMMAND cd ${CMAKE_SOURCE_DIR}/src/java && ./gradlew nativeCompile
    BYPRODUCTS ${GRAALVM_NATIVE_LIB_PATH}/${GRAALVM_NATIVE_LIB_NAME}
    COMMENT "Building GraalVM Native Image library -> ${GRAALVM_NATIVE_LIB_NAME}"
)

if(APPLE)
    add_custom_target(run-translate
        COMMAND DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/openjdk@21/21.0.7/libexec/openjdk.jdk/Contents/Home/lib/server:$ENV{DYLD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -translate "$ENV{QASM_FILE}"
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running QASM translation for file: $ENV{QASM_FILE}"
    )
elseif(UNIX AND NOT APPLE)
    add_custom_target(run-translate
        COMMAND LD_LIBRARY_PATH=/home/ark/.sdkman/candidates/java/current/lib/server:/home/ark/.sdkman/candidates/java/current/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -translate "$ENV{QASM_FILE}"
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running QASM translation for file: $ENV{QASM_FILE}"
    )
else()
    add_custom_target(run-translate
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -translate "$ENV{QASM_FILE}"
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running QASM translation for file: $ENV{QASM_FILE}"
    )
endif()

configure_file(
    ${CMAKE_SOURCE_DIR}/scripts/run.sh.template
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run.sh
    @ONLY
)

# 実行権限を付与
add_custom_target(generate-scripts
    COMMAND chmod +x ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run.sh
    DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/run.sh
    COMMENT "Setting executable permission for run.sh"
)

add_custom_target(all-with-scripts ALL
    DEPENDS qmdd_sim generate-scripts
    COMMENT "Building executable and generating scripts"
)