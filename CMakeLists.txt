cmake_minimum_required(VERSION 3.20)

# Project name and version
project(pENC VERSION 1.0 LANGUAGES CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the Windows SDK version
set(CMAKE_SYSTEM_VERSION 10.0.26100.0)

# Set the Python interpreter
find_package(Python3 3.13 REQUIRED COMPONENTS Interpreter Development)

# Find Cython package path using pip
execute_process(
    COMMAND python3.13 -m pip show cython
    OUTPUT_VARIABLE CYTHON_INFO
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCH "Location: ([^\n]+)" _ ${CYTHON_INFO})
set(CYTHON_PATH ${CMAKE_MATCH_1})

if(NOT CYTHON_PATH)
    message(FATAL_ERROR "Cython package not found")
endif()

# Set the OpenCL version
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=300)


# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR})
include_directories("C:/Program Files/OpenCL/include")
include_directories("C:/Program Files/clFFT/include")


# Link directories
link_directories("C:/Program Files/OpenCL/lib")
link_directories("C:/Program Files/clFFT/lib64/import")

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


# Add the mfcc_extractor library
add_library(mfcc_extractor SHARED mfcc_extractor.cpp mfcc_kernel.cl)
target_link_libraries(mfcc_extractor OpenCL.lib clFFT.lib)

# Set the output name for the DLL
set_target_properties(mfcc_extractor PROPERTIES OUTPUT_NAME "mfcc_extractor")

set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "" FORCE)

# Debug configuration for mfcc_extractor
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi /EHsc /LD /MDd")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} /DEBUG /INCREMENTAL")
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")

# Release configuration for mfcc_extractor
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /EHsc /LD /MD")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF /LTCG")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")



add_custom_target(Cython ALL
    COMMAND python3.13 setup.py build_ext --inplace --build-lib ${CMAKE_BINARY_DIR}/lib --build-temp ${CMAKE_BINARY_DIR}/temp
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)





# Set the default generator to Visual Studio 2022
if(NOT CMAKE_GENERATOR)
    set(CMAKE_GENERATOR "Visual Studio 17 2022")
    set(CMAKE_GENERATOR_PLATFORM "x64")
endif()
