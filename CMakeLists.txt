cmake_minimum_required(VERSION 3.18)
project(SplineTrajectoryExamples)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Python extension build (triggered by scikit-build-core or
# -DSPLINE_BUILD_PYTHON=ON)
# ============================================================
if(SKBUILD OR SPLINE_BUILD_PYTHON)
    find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)

    # Try system Eigen3 first; fall back to FetchContent
    find_package(Eigen3 QUIET)
    if(NOT TARGET Eigen3::Eigen)
        find_package(PkgConfig QUIET)
        if(PkgConfig_FOUND)
            pkg_check_modules(EIGEN3 eigen3)
        endif()
        if(NOT EIGEN3_FOUND AND NOT TARGET Eigen3::Eigen)
            include(FetchContent)
            FetchContent_Declare(eigen
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG 3.4.0
                GIT_SHALLOW TRUE
            )
            FetchContent_MakeAvailable(eigen)
        endif()
    endif()

    # pybind11 via FetchContent
    include(FetchContent)
    FetchContent_Declare(pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)

    python_add_library(_spline_trajectory MODULE
        bindings/module.cpp
        bindings/bind_splines_cubic.cpp
        bindings/bind_splines_quintic.cpp
        bindings/bind_splines_septic.cpp
        bindings/bind_optimizer_cubic.cpp
        bindings/bind_optimizer_quintic.cpp
        bindings/bind_optimizer_septic.cpp
    )

    target_include_directories(_spline_trajectory PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include/large_scale_traj_optimizer
        ${CMAKE_CURRENT_SOURCE_DIR}/bindings
    )

    target_link_libraries(_spline_trajectory PRIVATE
        pybind11::headers
        Eigen3::Eigen
        Python::NumPy
    )

    target_compile_options(_spline_trajectory PRIVATE
        $<$<CONFIG:Release>:-O3 -ffast-math -DNDEBUG>
        $<$<CONFIG:Debug>:-g -O0>
    )

    install(TARGETS _spline_trajectory DESTINATION spline_trajectory)
    return()
endif()

# ============================================================
# C++ tests / examples build (original)
# ============================================================

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Fallback for systems without proper Eigen3 CMake config
if(NOT TARGET Eigen3::Eigen)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(EIGEN3 REQUIRED eigen3)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/large_scale_traj_optimizer)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Performance test executables
add_executable(test_cubic_spline_vs_minco_nd test_cubic_spline_vs_minco_nd.cpp)
target_link_libraries(test_cubic_spline_vs_minco_nd Eigen3::Eigen)

add_executable(test_quintic_spline_vs_minco_nd test_quintic_spline_vs_minco_nd.cpp)
target_link_libraries(test_quintic_spline_vs_minco_nd Eigen3::Eigen)

add_executable(test_septic_spline_vs_minco_nd test_septic_spline_vs_minco_nd.cpp)
target_link_libraries(test_septic_spline_vs_minco_nd Eigen3::Eigen)

# Add new min_jerk comparison test
add_executable(test_with_min_jerk_3d test_with_min_jerk_3d.cpp)
target_link_libraries(test_with_min_jerk_3d Eigen3::Eigen)

# Add new min_snap comparison test
add_executable(test_with_min_snap_3d test_with_min_snap_3d.cpp)
target_link_libraries(test_with_min_snap_3d Eigen3::Eigen)

# Add gradient test
add_executable(test_Grad test_Grad.cpp)
target_link_libraries(test_Grad Eigen3::Eigen)

# Example executables
add_executable(basic_cubic_spline examples/basic_cubic_spline.cpp)
target_link_libraries(basic_cubic_spline Eigen3::Eigen)

add_executable(quintic_spline_comparison examples/quintic_spline_comparison.cpp)
target_link_libraries(quintic_spline_comparison Eigen3::Eigen)

add_executable(robot_trajectory_planning examples/robot_trajectory_planning.cpp)
target_link_libraries(robot_trajectory_planning Eigen3::Eigen)

# boundary condition grad test
add_executable(test_bc_grad test_bc_grad.cpp)
target_link_libraries(test_bc_grad Eigen3::Eigen)

# Cost gradient test
add_executable(test_cost_grad test_cost_grad.cpp)
target_link_libraries(test_cost_grad Eigen3::Eigen)

# PPolyND test
add_executable(test_ppolyND test_ppolyND.cpp)
target_link_libraries(test_ppolyND Eigen3::Eigen)

add_executable(test_spline_optimizer test_spline_optimizer.cpp)
target_link_libraries(test_spline_optimizer Eigen3::Eigen)

# Compiler flags
set(COMMON_COMPILE_FLAGS -Wall -Wextra)
set(DEBUG_FLAGS -g -O0)
set(RELEASE_FLAGS -O3 -march=native -ffast-math -DNDEBUG)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(test_cubic_spline_vs_minco_nd PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_quintic_spline_vs_minco_nd PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_septic_spline_vs_minco_nd PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_with_min_jerk_3d PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_with_min_snap_3d PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_Grad PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(basic_cubic_spline PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(quintic_spline_comparison PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(robot_trajectory_planning PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_bc_grad PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_cost_grad PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_ppolyND PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(test_spline_optimizer PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
else()
    target_compile_options(test_cubic_spline_vs_minco_nd PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_quintic_spline_vs_minco_nd PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_septic_spline_vs_minco_nd PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_with_min_jerk_3d PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_with_min_snap_3d PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_Grad PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(basic_cubic_spline PRIVATE ${COMMON_COMPILE_FLAGS} ${DEBUG_FLAGS})
    target_compile_options(quintic_spline_comparison PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(robot_trajectory_planning PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_bc_grad PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_cost_grad PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_ppolyND PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
    target_compile_options(test_spline_optimizer PRIVATE ${COMMON_COMPILE_FLAGS} ${RELEASE_FLAGS})
endif()

# Add chrono library for performance measurements
if(NOT WIN32)
    target_link_libraries(test_cubic_spline_vs_minco_nd pthread)
    target_link_libraries(test_quintic_spline_vs_minco_nd pthread)
    target_link_libraries(test_septic_spline_vs_minco_nd pthread)
    target_link_libraries(test_with_min_jerk_3d pthread)
    target_link_libraries(test_with_min_snap_3d pthread)
    target_link_libraries(test_Grad pthread)
    target_link_libraries(quintic_spline_comparison pthread)
    target_link_libraries(test_bc_grad pthread)
    target_link_libraries(test_cost_grad pthread)
    target_link_libraries(test_ppolyND pthread)
    target_link_libraries(test_spline_optimizer pthread)
endif()

# Create custom targets for easy building
add_custom_target(tests
    DEPENDS test_cubic_spline_vs_minco_nd test_quintic_spline_vs_minco_nd test_with_min_jerk_3d test_with_min_snap_3d test_septic_spline_vs_minco_nd test_Grad test_bc_grad test_cost_grad test_ppolyND test_spline_optimizer
    COMMENT "Building performance tests"
)

add_custom_target(examples
    DEPENDS basic_cubic_spline quintic_spline_comparison robot_trajectory_planning 
    COMMENT "Building examples"
)

add_custom_target(all_targets
    DEPENDS tests examples
    COMMENT "Building all targets"
)

# Status messages
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++标准: ${CMAKE_CXX_STANDARD}")
message(STATUS "项目名称: ${PROJECT_NAME}")
message(STATUS "可执行文件:")
message(STATUS "  - Performance Tests:")
message(STATUS "    * test_cubic_spline_vs_minco_nd")
message(STATUS "    * test_quintic_spline_vs_minco_nd")
message(STATUS "    * test_septic_spline_vs_minco_nd")
message(STATUS "    * test_with_min_jerk_3d")
message(STATUS "    * test_with_min_snap_3d")
message(STATUS "    * test_Grad")
message(STATUS "    * test_ppolyND")
message(STATUS "    * test_cost_grad")
message(STATUS "    * test_spline_optimizer")
message(STATUS "  - Examples:")
message(STATUS "    * basic_cubic_spline")
message(STATUS "    * quintic_spline_comparison") 
message(STATUS "    * robot_trajectory_planning")
message(STATUS "")
message(STATUS "构建命令:")
message(STATUS "  make tests     - 构建性能测试")
message(STATUS "  make examples  - 构建示例程序")
message(STATUS "  make all       - 构建所有目标")
