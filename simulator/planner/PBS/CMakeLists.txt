
cmake_minimum_required(VERSION 3.16)
project(pbs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options system filesystem)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

# Compile Jesse's modified PBS planner.
file(GLOB PBS_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

add_executable(pbs ${PBS_SOURCES})
target_include_directories(pbs PRIVATE inc)
target_link_libraries(pbs PRIVATE
    ${Boost_LIBRARIES}
    spdlog::spdlog
    rpc
    Threads::Threads
)

# Build the additional regression tests expected by the Dockerfile.
option(LSMART_BUILD_ROUTING_TESTS "Build PBS routing tests" OFF)

if(LSMART_BUILD_ROUTING_TESTS)
    set(TEST_SOURCES ${PBS_SOURCES})
    list(FILTER TEST_SOURCES EXCLUDE REGEX "driver\\.cpp$")

    foreach(test_name IN ITEMS routing_penalties route_search)
        add_executable(${test_name}_test
            tests/${test_name}.cpp
            ${TEST_SOURCES}
        )

        target_include_directories(${test_name}_test PRIVATE inc)
        target_link_libraries(${test_name}_test PRIVATE
            ${Boost_LIBRARIES}
            spdlog::spdlog
            rpc
            Threads::Threads
        )
    endforeach()
endif()

