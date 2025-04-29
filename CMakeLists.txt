# 指定运行此配置文件所需的 CMake 的最低版本
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# 指定项目的名称
project(petsc_petsc_mumps_test)

# 指定编译选项
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
# 展示所有编译单元所执行的指令 包括 Entering directory Leaving directory
set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
set(CMAKE_CXX_COMPILER "g++")
# 启用颜色输出，默认为ON
SET(CMAKE_COLOR_MAKEFILE ON)
# 把编译选项设置为了C++ 11
set(CMAKE_CXX_STANDARD 17)
# 构建中展示更详细的输出
SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(CMAKE_BUILD_TYPE Release)

# 默认的构建类型
IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Debug)
ENDIF()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# 指定生成文件路径
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 全局头文件包含
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    # ${Boost_INCLUDE_DIRS}
)

# 全局链接路径包含
link_directories(
    ${PROJECT_SOURCE_DIR}/lib

    # 写第三方动态库文件存放的路径
    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
)
# MPI 
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message(STATUS "MPI found, enabling support.")
    include_directories(${MPI_INCLUDE_PATH})
    # 不需要手动设置编译器，find_package(MPI) 会自动设置
    # set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
    # set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
endif()

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)
# 添加要编译的其他文件
add_subdirectory(src)
