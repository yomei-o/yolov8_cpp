cmake_minimum_required(VERSION 3.18)
project(yolov8cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LibTorch (CPU 2.5.1)
set(Torch_DIR "C:/prog/vc/third_party/libtorch/share/cmake/Torch" CACHE PATH "")
find_package(Torch REQUIRED)

add_executable(step1 step1_dfl_ciou.cpp)
target_link_libraries(step1 "${TORCH_LIBRARIES}")

add_executable(step2 step2_tal.cpp)
target_link_libraries(step2 "${TORCH_LIBRARIES}")

add_executable(step3a step3a_dfl_loss.cpp)
target_link_libraries(step3a "${TORCH_LIBRARIES}")

add_executable(step3b step3b_loss_forward.cpp)
target_link_libraries(step3b "${TORCH_LIBRARIES}")

add_executable(step3c step3c_loss_backward.cpp)
target_link_libraries(step3c "${TORCH_LIBRARIES}")

add_executable(step4 step4_train.cpp)
target_link_libraries(step4 "${TORCH_LIBRARIES}")

add_executable(step5 step5_train_yolov8n.cpp)
target_link_libraries(step5 "${TORCH_LIBRARIES}")

# Copy LibTorch DLLs next to the exe so it runs without PATH tweaks
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  foreach(tgt step1 step2 step3a step3b step3c step4 step5)
    add_custom_command(TARGET ${tgt} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:${tgt}>)
  endforeach()
endif()
