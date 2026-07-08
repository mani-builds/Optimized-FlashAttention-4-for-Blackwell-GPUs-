
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(flash_pytorch_extension LANGUAGES CXX CUDA)
find_package(Torch REQUIRED)
# 2. Find Python and its development headers (Python.h)
# This automatically searches your active Conda environment
find_package(Python3 COMPONENTS Development REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


add_executable(flash_v1 flash_v1.cu)
target_link_libraries(flash_v1 "${TORCH_LIBRARIES}")
set_property(TARGET flash_v1 PROPERTY CXX_STANDARD 20)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET flash_v1
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:flash_v1>)
endif (MSVC)

# # 1. Tell CMake where to find LibTorch
# # set(CMAKE_PREFIX_PATH "/home/mani/libtorch")
# find_package(Torch REQUIRED)

# # 2. Find Python and its development headers (Python.h)
# # This automatically searches your active Conda environment
# find_package(Python3 COMPONENTS Development REQUIRED)

# # 3. Define your extension library
# add_library(flash_v1 SHARED flash_v1.cu)

# # 4. Link LibTorch, Python, and their headers to your target
# target_link_libraries(flash_v1
#     PRIVATE
#     "${TORCH_LIBRARIES}"
#     Python3::Python
# )

# # 5. Set required C++ standard
# set_property(TARGET flash_v1 PROPERTY CXX_STANDARD 17)
