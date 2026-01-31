cmake_minimum_required(VERSION 3.15...3.31)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# =============================================================================
# Build libcdp (the C library)
# =============================================================================
set(LIBCDP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libcdp")

# libcdp source files
set(LIBCDP_SOURCES
    ${LIBCDP_DIR}/src/error.c
    ${LIBCDP_DIR}/src/context.c
    ${LIBCDP_DIR}/src/buffer.c
    ${LIBCDP_DIR}/src/gain.c
)

# Build libcdp as a static library
add_library(cdp STATIC ${LIBCDP_SOURCES})
target_include_directories(cdp PUBLIC ${LIBCDP_DIR}/include)
if(UNIX)
    target_link_libraries(cdp PUBLIC m)
endif()

# =============================================================================
# Build Cython extension
# =============================================================================
add_custom_command(
    OUTPUT _core.c
    COMMAND Python::Interpreter -m cython
        "${CMAKE_CURRENT_SOURCE_DIR}/src/pycdp/_core.pyx" --output-file _core.c
    DEPENDS src/pycdp/_core.pyx
)

python_add_library(_core MODULE _core.c WITH_SOABI)

# Link against libcdp
target_link_libraries(_core PRIVATE cdp)
target_include_directories(_core PRIVATE ${LIBCDP_DIR}/include)

if(MSVC)
    target_compile_options(_core PRIVATE /W4)
else()
    target_compile_options(_core PRIVATE -Wall -Wextra)
endif()

install(TARGETS _core DESTINATION pycdp)
