# CMakeLists.txt for CDP as a library
# This builds CDP's processing code as a static library with memory I/O shim

cmake_minimum_required(VERSION 3.15)
project(cdp_lib C)

set(CMAKE_C_STANDARD 99)

# CDP source directory
set(CDP_DEV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../dev")

# Compiler flags
if(APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Dunix -D__MAC__ -DMAC -DENABLE_PVX -DCDP_LIBRARY_MODE")
elseif(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -Dunix -Dlinux -D_X86_ -DENABLE_PVX -DCDP_LIBRARY_MODE")
endif()

# Include directories
include_directories(
    ${CDP_DEV_DIR}/newinclude
    ${CDP_DEV_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# =============================================================================
# SFSYS Library (with shim support)
# =============================================================================

set(SFSYS_SRCS
    ${CDP_DEV_DIR}/newsfsys/sfsys.c
    ${CDP_DEV_DIR}/newsfsys/snd.c
    ${CDP_DEV_DIR}/newsfsys/sfdir.c
    ${CDP_DEV_DIR}/newsfsys/osbind.c
    ${CDP_DEV_DIR}/newsfsys/props.c
    ${CDP_DEV_DIR}/newsfsys/ieee80.c
)

# Our shim and wrapper layer
set(SHIM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/cdp_shim.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cdp_io_redirect.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cdp_lib.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cdp_spectral.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cdp_envelope.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cdp_distort.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cdp_reverb.c
    ${CMAKE_CURRENT_SOURCE_DIR}/cdp_granular.c
)

# =============================================================================
# CDP2K Core Library
# =============================================================================

set(CDP2K_SRCS
    ${CDP_DEV_DIR}/cdparse/cdparse.c
    ${CDP_DEV_DIR}/cdp2k/dzsetup.c
    ${CDP_DEV_DIR}/cdp2k/formantsg.c
    ${CDP_DEV_DIR}/cdp2k/mainfuncs.c
    ${CDP_DEV_DIR}/cdp2k/parstruct.c
    ${CDP_DEV_DIR}/cdp2k/readdata.c
    ${CDP_DEV_DIR}/cdp2k/readfiles.c
    ${CDP_DEV_DIR}/cdp2k/special.c
    ${CDP_DEV_DIR}/cdp2k/tkinput.c
    ${CDP_DEV_DIR}/cdp2k/tklib1.c
    ${CDP_DEV_DIR}/cdp2k/tklib3.c
    ${CDP_DEV_DIR}/cdp2k/validate.c
    ${CDP_DEV_DIR}/cdp2k/writedata.c
)

# =============================================================================
# PVXIO2 Library (for spectral file handling)
# =============================================================================

set(PVXIO2_SRCS
    ${CDP_DEV_DIR}/pvxio2/pvfileio.c
)

# =============================================================================
# Processing Libraries (without main.c)
# =============================================================================

# Stretch
set(STRETCH_SRCS
    ${CDP_DEV_DIR}/stretch/ap_stretch.c
    ${CDP_DEV_DIR}/stretch/stretch.c
)

# Blur
set(BLUR_SRCS
    ${CDP_DEV_DIR}/blur/ap_blur.c
    ${CDP_DEV_DIR}/blur/blur.c
)

# Phase vocoder
set(PV_SRCS
    ${CDP_DEV_DIR}/pv/ap_pvoc.c
    ${CDP_DEV_DIR}/pv/pvoc.c
    ${CDP_DEV_DIR}/pv/mxfft.c
)

# Modify (loudness, speed, etc.)
set(MODIFY_SRCS
    ${CDP_DEV_DIR}/modify/ap_modify.c
    ${CDP_DEV_DIR}/modify/radical.c
    ${CDP_DEV_DIR}/modify/gain.c
    ${CDP_DEV_DIR}/modify/pan.c
    ${CDP_DEV_DIR}/modify/strans.c
    ${CDP_DEV_DIR}/modify/delay.c
    ${CDP_DEV_DIR}/modify/granula1.c
    ${CDP_DEV_DIR}/modify/brapcon.c
)

# =============================================================================
# Build static library
# =============================================================================

add_library(cdp_processing STATIC
    ${SHIM_SRCS}
    ${CDP2K_SRCS}
    ${STRETCH_SRCS}
    ${BLUR_SRCS}
    ${PV_SRCS}
    ${MODIFY_SRCS}
)

# Link against sfsys (we'll build a modified version)
add_library(cdp_sfsys STATIC ${SFSYS_SRCS})

# PVXIO2
add_library(cdp_pvxio2 STATIC ${PVXIO2_SRCS})

# Combined library target
add_library(cdp_all INTERFACE)
target_link_libraries(cdp_all INTERFACE cdp_processing cdp_sfsys cdp_pvxio2)

# Test program
add_executable(test_cdp_lib test_cdp_lib.c)
target_link_libraries(test_cdp_lib cdp_processing cdp_sfsys cdp_pvxio2 m)
