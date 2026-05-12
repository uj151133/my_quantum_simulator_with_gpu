cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

project(Qmdd_Sim VERSION 0.1.0)

# ビルドタイプをReleaseに設定（最適化を常に有効化）
set(CMAKE_BUILD_TYPE Release)

set(GRAALVM_NATIVE_LIB_PATH "${CMAKE_SOURCE_DIR}/src/java/build/native/nativeCompile")
set(GRAALVM_NATIVE_LIB_SEARCH_NAME "operationcache")

if(APPLE)
    set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "26.0")
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

# 最適化フラグ（精度を保ちつつ高速化）
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")

set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_UNVERSIONED_NAMES FIRST)

find_package(Python3 COMPONENTS Interpreter Development.Module Development.Embed REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

message(STATUS "Using Python: ${Python3_EXECUTABLE} (${Python3_VERSION})")

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import site,sys;print(site.getusersitepackages())"
  OUTPUT_VARIABLE PY_USER_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

include_directories(${Python3_INCLUDE_DIRS})

include(cmake/Dependencies.cmake)

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
        /usr/local/lib
        /usr/lib
        /usr/lib/x86_64-linux-gnu  # Ubuntu特有のパスを追加
        REQUIRED
    )
    find_path(GMP_INCLUDE_DIRS
        NAMES gmp.h
        PATHS
        /usr/local/include
        /usr/include
        REQUIRED
    )
    if(GMP_LIB AND GMP_INCLUDE_DIRS)
        include_directories(${GMP_INCLUDE_DIRS})
    else()
        message(FATAL_ERROR "GMP library not found. Please install libgmp-dev")
    endif()
    find_path(ANTLR4_RUNTIME_INCLUDE_DIR
        NAMES antlr4-runtime.h
        PATHS /usr/local/include /usr/include
        PATH_SUFFIXES antlr4-runtime
    )
    find_library(ANTLR4_RUNTIME_LIB
        NAMES antlr4-runtime
        PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
    )
    if(NOT ANTLR4_RUNTIME_INCLUDE_DIR)
        message(FATAL_ERROR "antlr4-runtime headers not found. Install libantlr4-runtime 4.13.1+ or set ANTLR4_RUNTIME_INCLUDE_DIR")
    endif()
    include_directories(${ANTLR4_RUNTIME_INCLUDE_DIR})
endif()

if(ANTLR4_RUNTIME_INCLUDE_DIR AND ANTLR4_RUNTIME_LIB)
    add_library(antlr4_runtime UNKNOWN IMPORTED GLOBAL)
    set_target_properties(antlr4_runtime PROPERTIES
        IMPORTED_LOCATION "${ANTLR4_RUNTIME_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${ANTLR4_RUNTIME_INCLUDE_DIR}"
    )
else()
    message(FATAL_ERROR "antlr4-runtime not configured correctly")
endif()

if(APPLE)
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS /opt/homebrew/include /usr/local/include /usr/include
        PATH_SUFFIXES onnxruntime
    )
    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        HINTS /opt/homebrew/lib /usr/local/lib /usr/lib
    )
else()
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            ${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.19.0/include
            ~/onnxruntime-linux-x64-1.19.0/include
            /usr/local/include
            /usr/include
        PATH_SUFFIXES onnxruntime
    )

    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
            ${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.19.0/lib
            ~/onnxruntime-linux-x64-1.19.0/lib
            /usr/local/lib
            /usr/lib
            /usr/lib/x86_64-linux-gnu
    )
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIB}")
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
else()
    message(WARNING "ONNX Runtime not found. SchedulerONNX will fall back to identity perm.")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include_directories(SYSTEM 
    ${Boost_INCLUDE_DIRS}
    ${BOOST_ROOT}/include
    ${Python3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    /opt/homebrew/include
    $ENV{JAVA_HOME}/include
    $ENV{JAVA_HOME}/include/darwin
    ${CMAKE_SOURCE_DIR}/src
    ${GRAALVM_NATIVE_LIB_PATH}
)

link_directories(
    ${Python3_LIBRARY_DIRS}
    ${YAML_CPP_LIB_DIR}
    /opt/homebrew/lib
    ${GRAALVM_NATIVE_LIB_PATH}
)

if(APPLE)
    add_compile_definitions(CLOCK_MONOTONIC_ID=6)
else()
    add_compile_definitions(CLOCK_MONOTONIC_ID=1)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${BOOST_ROOT}/include")

# 出力ディレクトリを親ディレクトリに設定
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/libs/__pycache__)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

include_directories(src)

enable_language(C)
enable_language(ASM)
enable_language(Fortran)
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

# === GPU backend auto-detect (NO option) ===
set(GPU_BACKEND_SRC "")
if(APPLE)
    add_compile_definitions(USE_METAL)
    set(GPU_BACKEND_SRC src/modules/Heian.mm)
    set(METAL_SRC ${CMAKE_SOURCE_DIR}/src/modules/Heian.metal)
    set(METAL_AIR ${CMAKE_BINARY_DIR}/Heian.air)
    set(METAL_LIB ${CMAKE_BINARY_DIR}/Heian.metallib)

    add_custom_command(
        OUTPUT ${METAL_AIR}
        COMMAND xcrun -sdk macosx metal -c ${METAL_SRC} -o ${METAL_AIR}
        DEPENDS ${METAL_SRC}
        COMMENT "Compiling Metal shader"
    )

    add_custom_command(
        OUTPUT ${METAL_LIB}
        COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
        DEPENDS ${METAL_AIR}
        COMMENT "Linking Metal library"
    )

    add_custom_target(metal_shaders ALL DEPENDS ${METAL_LIB})
else()
    find_package(CUDAToolkit)

    if(CUDAToolkit_FOUND)
        # nvcc を明示的に探して、CMake の CUDA compiler として設定
        find_program(NVCC_PATH NAMES nvcc HINTS /usr/local/cuda/bin ENV CUDA_HOME PATH_SUFFIXES bin)
        if(NOT NVCC_PATH)
            message(FATAL_ERROR "CUDAToolkit は見つかったが nvcc が見つかりません。CMAKE_CUDA_COMPILER を指定してください。")
        endif()

        if(NOT CMAKE_CUDA_COMPILER)
            set(CMAKE_CUDA_COMPILER "${NVCC_PATH}" CACHE FILEPATH "CUDA compiler" FORCE)
        endif()

        # 自動検出失敗対策（必要なら 86/89 など固定値に変更）
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
        endif()

        enable_language(CUDA)
        add_compile_definitions(USE_CUDA)
        set(GPU_BACKEND_SRC src/modules/Heian.cu)
    else()
        message(FATAL_ERROR "No GPU backend available (Metal/CUDA not found)")
    endif()
endif()

configure_file(${CMAKE_SOURCE_DIR}/config.yaml ${CMAKE_BINARY_DIR}/config.yaml COPYONLY)
configure_file(${CMAKE_SOURCE_DIR}/cmake/qmdd_project.pth.in ${CMAKE_BINARY_DIR}/qmdd_project.pth @ONLY)
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
    
    if(EXISTS "${GRAALVM_NATIVE_LIB_PATH}/${GRAALVM_NATIVE_LIB_NAME}")
        message(STATUS "File exists but not found by find_library, using direct path")
        set(GRAALVM_NATIVE_LIB "${GRAALVM_NATIVE_LIB_PATH}/${GRAALVM_NATIVE_LIB_NAME}")
    endif()
endif()

set(OPENQASM3_GRAMMAR_DIR ${CMAKE_SOURCE_DIR}/src/translator/OpenQASM3)
set(OPENQASM3_GRAMMAR_FILE ${OPENQASM3_GRAMMAR_DIR}/OpenQASM3.g4)
set(OPENQASM3_GEN_DIR ${OPENQASM3_GRAMMAR_DIR}/gen)

if(NOT ANTLR_JAR)
    if(DEFINED ENV{ANTLR_JAR} AND EXISTS "$ENV{ANTLR_JAR}")
        set(ANTLR_JAR "$ENV{ANTLR_JAR}")
    elseif(EXISTS "/opt/homebrew/Cellar/antlr/4.13.2/antlr-4.13.2-complete.jar")
        set(ANTLR_JAR "/opt/homebrew/Cellar/antlr/4.13.2/antlr-4.13.2-complete.jar")
    elseif(EXISTS "/usr/share/java/antlr-4.13.2-complete.jar")
        set(ANTLR_JAR "/usr/share/java/antlr-4.13.2-complete.jar")
    elseif(EXISTS "/usr/share/java/antlr-4.13.1-complete.jar")
        set(ANTLR_JAR "/usr/share/java/antlr-4.13.1-complete.jar")
    elseif(EXISTS "/usr/local/share/java/antlr-4.13.2-complete.jar")
        set(ANTLR_JAR "/usr/local/share/java/antlr-4.13.2-complete.jar")
    endif()
endif()

if(NOT ANTLR_JAR OR NOT EXISTS "${ANTLR_JAR}")
    message(FATAL_ERROR "ANTLR_JAR not found. export ANTLR_JAR=/opt/homebrew/Cellar/antlr/4.13.2/antlr-4.13.2-complete.jar か -DANTLR_JAR=指定")
else()
    message(STATUS "Using ANTLR JAR: ${ANTLR_JAR}")
endif()

if(NOT EXISTS "${ANTLR_JAR}")
    message(FATAL_ERROR "ANTLR_JAR not found. 指定してください: -DANTLR_JAR=/path/to/antlr-4.x-complete.jar (現在: '${ANTLR_JAR}')")
else()
    message(STATUS "Using ANTLR JAR: ${ANTLR_JAR}")
endif()

find_package(Java COMPONENTS Runtime REQUIRED)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENQASM3_GEN_DIR}
)
execute_process(
    COMMAND ${Java_JAVA_EXECUTABLE} -jar ${ANTLR_JAR} -Dlanguage=Cpp -visitor -no-listener -o ${OPENQASM3_GEN_DIR} ${OPENQASM3_GRAMMAR_FILE}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    RESULT_VARIABLE ANTLR_CFG_RES
)
if(NOT ANTLR_CFG_RES EQUAL 0)
    message(FATAL_ERROR "ANTLR generate (configure-time) failed")
endif()

set(OPENQASM3_GEN_SRCS
    ${OPENQASM3_GEN_DIR}/OpenQASM3Lexer.cpp
    ${OPENQASM3_GEN_DIR}/OpenQASM3Parser.cpp
    ${OPENQASM3_GEN_DIR}/OpenQASM3BaseVisitor.cpp
    ${OPENQASM3_GEN_DIR}/OpenQASM3Visitor.cpp
)

add_custom_command(
    OUTPUT ${OPENQASM3_GEN_SRCS}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${OPENQASM3_GEN_DIR}
    COMMAND ${Java_JAVA_EXECUTABLE} -jar ${ANTLR_JAR} -Dlanguage=Cpp -visitor -no-listener -o ${OPENQASM3_GEN_DIR} ${OPENQASM3_GRAMMAR_FILE}
    DEPENDS ${OPENQASM3_GRAMMAR_FILE}
    COMMENT "Generating OpenQASM3 (ANTLR)"
    VERBATIM
)
add_custom_target(generate_qasm3 DEPENDS
    ${OPENQASM3_GEN_DIR}/OpenQASM3Lexer.cpp
    ${OPENQASM3_GEN_DIR}/OpenQASM3Parser.cpp
    ${OPENQASM3_GEN_DIR}/OpenQASM3BaseVisitor.cpp
    ${OPENQASM3_GEN_DIR}/OpenQASM3Visitor.cpp
)

file(GLOB SOURCES CONFIGURE_DEPENDS "src/models/*.cpp" "src/common/*.cpp" "src/modules/*.cpp" "src/test/*/*.cpp" "src/translator/*/*.cpp" "src/opt/*.cpp")
file(GLOB OPENQASM3_GEN_SRC CONFIGURE_DEPENDS "${OPENQASM3_GEN_DIR}/*.cpp")

add_executable(qmdd_sim main.cpp ${SOURCES} ${C_SRC} ${ASM_SRC} ${OPENQASM3_GEN_SRC} src/common/timer.f90)
add_dependencies(qmdd_sim copy_config build_java_lib generate_qasm3)
target_include_directories(qmdd_sim PRIVATE ${GSL_INCLUDE_DIRS} ${OPENQASM3_GEN_DIR} ${ANTLR4_RUNTIME_INCLUDE_DIR})
target_sources(qmdd_sim PRIVATE ${GPU_BACKEND_SRC})
target_link_libraries(qmdd_sim PRIVATE
    pthread
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    ${Boost_LIBRARIES}
    GSL::gsl
    ${GMP_LIB}
    Threads::Threads
    xxhash
    antlr4_runtime
    $<$<BOOL:${GRAALVM_NATIVE_LIB}>:${GRAALVM_NATIVE_LIB}>
    $<$<BOOL:${ONNXRUNTIME_LIB}>:${ONNXRUNTIME_LIB}>
)

if(APPLE)
    target_link_libraries(qmdd_sim PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreFoundation"
    )
endif()

if(APPLE AND ONNXRUNTIME_LIB)
    get_filename_component(ONNX_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)
    set_target_properties(qmdd_sim PROPERTIES
        BUILD_RPATH "${ONNX_LIB_DIR}"
        INSTALL_RPATH "@loader_path;${ONNX_LIB_DIR}"
    )
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(qmdd_sim PROPERTIES BUILD_RPATH "/usr/local/lib")
endif()

if(APPLE)
    set(LIB_PATH_PREFIX "DYLD_LIBRARY_PATH=/opt/homebrew/Cellar/openjdk@21/21.0.7/libexec/openjdk.jdk/Contents/Home/lib/server:\$DYLD_LIBRARY_PATH")
elseif(UNIX AND NOT APPLE)
    set(LIB_PATH_PREFIX "LD_LIBRARY_PATH=/home/ark/.sdkman/candidates/java/current/lib/server:/home/ark/.sdkman/candidates/java/current/lib:\$LD_LIBRARY_PATH")
else()
    set(LIB_PATH_PREFIX "")
endif()

# === Python拡張モジュール: qmdd_core ===

file(GLOB CORE_SOURCES
    "src/models/*.cpp"
    "src/common/*.cpp"
    "src/modules/*.cpp"
    "src/bindings/*.cpp"
    "src/opt/*.cpp"
)

add_library(qmdd_core MODULE
    ${CORE_SOURCES}
    ${C_SRC}
    ${ASM_SRC}
)

if(APPLE)
    target_compile_definitions(qmdd_sim PRIVATE USE_METAL)
    target_compile_definitions(qmdd_core PRIVATE USE_METAL)
endif()

if(TARGET metal_shaders)
    add_dependencies(qmdd_sim metal_shaders)
    add_dependencies(qmdd_core metal_shaders)
endif()

set_target_properties(qmdd_core PROPERTIES
    PREFIX ""
    OUTPUT_NAME "qmdd_core"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/AI"
)

target_include_directories(qmdd_core PRIVATE
    ${GSL_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    ${Python3_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
)

target_sources(qmdd_core PRIVATE ${GPU_BACKEND_SRC})
target_link_libraries(qmdd_core PRIVATE
    pybind11::module
    pthread
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    ${Boost_LIBRARIES}
    GSL::gsl
    ${GMP_LIB}
    Threads::Threads
    xxhash
    $<$<BOOL:${GRAALVM_NATIVE_LIB}>:${GRAALVM_NATIVE_LIB}>
    $<$<BOOL:${ONNXRUNTIME_LIB}>:${ONNXRUNTIME_LIB}>
)

if(APPLE)
    target_link_libraries(qmdd_core PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework CoreFoundation"
    )
endif()

if(APPLE AND ONNXRUNTIME_LIB)
    get_filename_component(ONNX_LIB_DIR "${ONNXRUNTIME_LIB}" DIRECTORY)
    set_target_properties(qmdd_core PROPERTIES
        BUILD_RPATH "${ONNX_LIB_DIR}"
        INSTALL_RPATH "@loader_path;${ONNX_LIB_DIR}"
    )
endif()

if(UNIX AND NOT APPLE)
    set_target_properties(qmdd_core PROPERTIES BUILD_RPATH "/usr/local/lib")
endif()

# make run ターゲットの追加（通常モード）
if(APPLE)
    add_custom_target(run
        COMMAND DYLD_LIBRARY_PATH=$ENV{JAVA_HOME}/lib/server:$ENV{DYLD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running the application in standalone mode..."
    )
elseif(UNIX AND NOT APPLE)
    add_custom_target(run
        COMMAND LD_LIBRARY_PATH=$ENV{JAVA_HOME}/lib/server:$ENV{JAVA_HOME}/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim
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
        COMMAND DYLD_LIBRARY_PATH=$ENV{JAVA_HOME}/lib/server:$ENV{DYLD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -s
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running the application in IPC server mode for GUI communication..."
    )
elseif(UNIX AND NOT APPLE)
    add_custom_target(run-server
        COMMAND LD_LIBRARY_PATH=$ENV{JAVA_HOME}/lib/server:$ENV{JAVA_HOME}/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -s
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
        COMMAND DYLD_LIBRARY_PATH=$ENV{JAVA_HOME}/lib/server:$ENV{DYLD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -t "$ENV{QASM_FILE}"
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running QASM translation for file: $ENV{QASM_FILE}"
    )
elseif(UNIX AND NOT APPLE)
    add_custom_target(run-translate
        COMMAND LD_LIBRARY_PATH=$ENV{JAVA_HOME}/lib/server:$ENV{JAVA_HOME}/lib:$ENV{LD_LIBRARY_PATH} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -t "$ENV{QASM_FILE}"
        DEPENDS qmdd_sim
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        COMMENT "Running QASM translation for file: $ENV{QASM_FILE}"
    )
else()
    add_custom_target(run-translate
        COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/qmdd_sim -t "$ENV{QASM_FILE}"
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

add_custom_target(install_usersite_pth ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory "${PY_USER_SITE}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_BINARY_DIR}/qmdd_project.pth"
          "${PY_USER_SITE}/qmdd_project.pth"
  BYPRODUCTS "${PY_USER_SITE}/qmdd_project.pth"
  COMMENT "Installing Python .pth into ${PY_USER_SITE}"
)