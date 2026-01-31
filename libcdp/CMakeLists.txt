# CDP Library - CMake Build Configuration
cmake_minimum_required(VERSION 3.5)
project(libcdp VERSION 0.1.0 LANGUAGES C)

# C standard
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Build type default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Source files
set(CDP_SOURCES
    src/error.c
    src/context.c
    src/buffer.c
    src/gain.c
    src/io.c
    src/channel.c
    src/mix.c
    src/spatial.c
    src/utils.c
)

# Header files (for IDE integration)
set(CDP_HEADERS
    include/cdp.h
    include/cdp_error.h
    include/cdp_types.h
)

# Build as static library (can also build shared)
add_library(cdp STATIC ${CDP_SOURCES} ${CDP_HEADERS})

# Include directories
target_include_directories(cdp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link math library on Unix
if(UNIX)
    target_link_libraries(cdp PUBLIC m)
endif()

# Build shared library option
option(BUILD_SHARED_LIBS "Build shared library" OFF)
if(BUILD_SHARED_LIBS)
    add_library(cdp_shared SHARED ${CDP_SOURCES})
    target_include_directories(cdp_shared
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    if(UNIX)
        target_link_libraries(cdp_shared PUBLIC m)
    endif()
    set_target_properties(cdp_shared PROPERTIES OUTPUT_NAME cdp)
endif()

# Tests
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_cdp tests/test_cdp.c)
    target_link_libraries(test_cdp cdp)
    add_test(NAME test_cdp COMMAND test_cdp)
endif()

# Installation
include(GNUInstallDirs)
install(TARGETS cdp
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES ${CDP_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cdp)

# pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cdp.pc.in
    ${CMAKE_CURRENT_BINARY_DIR}/cdp.pc
    @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cdp.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)
