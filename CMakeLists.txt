cmake_minimum_required(VERSION 3.20)
project(matrix LANGUAGES CXX OBJCXX)

add_library(matrix STATIC
    Matrix.mm
    Mods/Utils.mm
)

target_include_directories(matrix PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_features(matrix PUBLIC cxx_std_23)

target_compile_options(matrix PUBLIC
    -fmodules
    -fcxx-modules
    -fmodule-map-file=${CMAKE_CURRENT_SOURCE_DIR}/module.modulemap
)

find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)
find_library(IMAGEIO_FRAMEWORK ImageIO REQUIRED)
find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED) # resample()'s vImage CPU fast path

target_link_libraries(matrix PUBLIC
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
    ${COREGRAPHICS_FRAMEWORK}
    ${IMAGEIO_FRAMEWORK}
    ${ACCELERATE_FRAMEWORK}
)

# --- The compute kernels ---
# GPUManager loads its kernels from a default.metallib. In an app, Xcode compiles ComputeShaders/*.metal
# into the bundle's; a plain CMake executable has no bundle, so this compiles the same sources into
# default.metallib in the build folder with the real `xcrun metal`/`xcrun metallib` toolchain, and
# hands its path to GPUManager as MATRIX_METALLIB_PATH. GPUManager tries $MATRIX_METALLIB and a
# default.metallib next to the executable first, so a project with kernels of its own (gpu_func) can
# build one metallib with matrix's and its own and put it there.
find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

file(GLOB METAL_SHADER_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/ComputeShaders/*.metal)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/metal_air)

set(METAL_AIR_FILES "")
foreach(metal_src ${METAL_SHADER_SOURCES})
    get_filename_component(metal_name ${metal_src} NAME_WE)
    set(air_file ${CMAKE_CURRENT_BINARY_DIR}/metal_air/${metal_name}.air)
    add_custom_command(
        OUTPUT ${air_file}
        COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metal -c ${metal_src} -I ${CMAKE_CURRENT_SOURCE_DIR}/ComputeShaders -o ${air_file}
        DEPENDS ${metal_src}
        COMMENT "metal: compiling ${metal_name}.metal"
        VERBATIM
    )
    list(APPEND METAL_AIR_FILES ${air_file})
endforeach()

set(DEFAULT_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/default.metallib)
add_custom_command(
    OUTPUT ${DEFAULT_METALLIB}
    COMMAND ${XCRUN_EXECUTABLE} -sdk macosx metallib ${METAL_AIR_FILES} -o ${DEFAULT_METALLIB}
    DEPENDS ${METAL_AIR_FILES}
    COMMENT "metallib: linking default.metallib"
    VERBATIM
)
add_custom_target(matrix_metallib ALL DEPENDS ${DEFAULT_METALLIB})
add_dependencies(matrix matrix_metallib)
target_compile_definitions(matrix PUBLIC MATRIX_METALLIB_PATH="${DEFAULT_METALLIB}")
