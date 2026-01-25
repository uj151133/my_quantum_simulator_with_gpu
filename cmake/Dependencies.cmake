find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Boost ${BOOST_MIN_VERSION} REQUIRED COMPONENTS
    thread
    fiber
    context
    filesystem
    program_options
)
find_package(Threads REQUIRED)
find_package(GSL REQUIRED)
