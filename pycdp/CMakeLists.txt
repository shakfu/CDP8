cmake_minimum_required(VERSION 3.15...3.31)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# =============================================================================
# Build libcdp (the C library)
# =============================================================================
set(LIBCDP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../libcdp")
set(CDP_DEV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dev")

# libcdp source files
set(LIBCDP_SOURCES
    ${LIBCDP_DIR}/src/error.c
    ${LIBCDP_DIR}/src/context.c
    ${LIBCDP_DIR}/src/buffer.c
    ${LIBCDP_DIR}/src/gain.c
    ${LIBCDP_DIR}/src/io.c
    ${LIBCDP_DIR}/src/channel.c
    ${LIBCDP_DIR}/src/mix.c
    ${LIBCDP_DIR}/src/spatial.c
    ${LIBCDP_DIR}/src/utils.c
)

# Build libcdp as a static library
add_library(cdp STATIC ${LIBCDP_SOURCES})
target_include_directories(cdp PUBLIC ${LIBCDP_DIR}/include)
if(UNIX)
    target_link_libraries(cdp PUBLIC m)
endif()

# =============================================================================
# Build CDP processing library (native spectral operations)
# =============================================================================

# CDP library wrapper sources
set(CDP_LIB_SOURCES
    ${LIBCDP_DIR}/cdp_lib/cdp_lib.c
    ${LIBCDP_DIR}/cdp_lib/cdp_spectral.c
    ${LIBCDP_DIR}/cdp_lib/cdp_io_redirect.c
    ${LIBCDP_DIR}/cdp_lib/cdp_envelope.c
    ${LIBCDP_DIR}/cdp_lib/cdp_distort.c
    ${LIBCDP_DIR}/cdp_lib/cdp_reverb.c
    ${LIBCDP_DIR}/cdp_lib/cdp_granular.c
    ${LIBCDP_DIR}/src/cdp_shim.c
)

# CDP dev sources needed (FFT)
set(CDP_FFT_SOURCES
    ${CDP_DEV_DIR}/pv/mxfft.c
)

# Build as static library
add_library(cdp_lib STATIC ${CDP_LIB_SOURCES} ${CDP_FFT_SOURCES})
target_include_directories(cdp_lib PUBLIC
    ${LIBCDP_DIR}/cdp_lib
    ${LIBCDP_DIR}/src
    ${CDP_DEV_DIR}/newinclude
    ${CDP_DEV_DIR}/include
)
if(APPLE)
    target_compile_definitions(cdp_lib PRIVATE unix __MAC__ MAC)
elseif(UNIX)
    target_compile_definitions(cdp_lib PRIVATE unix linux _X86_)
endif()
if(UNIX)
    target_link_libraries(cdp_lib PUBLIC m)
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

# Link against libcdp and cdp_lib
target_link_libraries(_core PRIVATE cdp cdp_lib)
target_include_directories(_core PRIVATE
    ${LIBCDP_DIR}/include
    ${LIBCDP_DIR}/cdp_lib
)

if(MSVC)
    target_compile_options(_core PRIVATE /W4)
else()
    target_compile_options(_core PRIVATE -Wall -Wextra)
endif()

install(TARGETS _core DESTINATION pycdp)
