cmake_minimum_required(VERSION 3.10)

project(rk3568_camera_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(RKNN_INCLUDE_DIR "/userdata/rknn_model_zoo_ppocr_min/3rdparty/rknpu2/include" CACHE PATH "RKNN headers")
set(RKNN_LIBRARY "/userdata/rknn_model_zoo_ppocr_min/3rdparty/rknpu2/Linux/aarch64/librknnrt.so" CACHE FILEPATH "RKNN runtime")
set(RGA_INCLUDE_DIR "/userdata/rknn_model_zoo_ppocr_min/3rdparty/librga/include" CACHE PATH "RGA headers")
set(RGA_LIBRARY "/userdata/rknn_model_zoo_ppocr_min/3rdparty/librga/Linux/aarch64/librga.so" CACHE FILEPATH "RGA runtime")
set(PPOCR_VENDOR_ROOT "/userdata/rknn_model_zoo_ppocr_min" CACHE PATH "Rockchip PPOCR example source")
set(TURBOJPEG_LIBRARY "${PPOCR_VENDOR_ROOT}/3rdparty/jpeg_turbo/Linux/aarch64/libturbojpeg.a" CACHE FILEPATH "TurboJPEG library")

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(Threads REQUIRED)

add_library(rk3568_ppocr_vendor STATIC
    ${PPOCR_VENDOR_ROOT}/examples/PPOCR/PPOCR-System/cpp/clipper.cc
    ${PPOCR_VENDOR_ROOT}/examples/PPOCR/PPOCR-System/cpp/postprocess.cc
    vendor/rockchip_ppocr/ppocr_system.cc
    ${PPOCR_VENDOR_ROOT}/utils/file_utils.c
    ${PPOCR_VENDOR_ROOT}/utils/image_utils.c
)
target_include_directories(rk3568_ppocr_vendor PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/rockchip_ppocr
    ${PPOCR_VENDOR_ROOT}/examples/PPOCR/PPOCR-System/cpp
    ${PPOCR_VENDOR_ROOT}/utils
    ${PPOCR_VENDOR_ROOT}/3rdparty/jpeg_turbo/include
    ${PPOCR_VENDOR_ROOT}/3rdparty/librga/include
    ${PPOCR_VENDOR_ROOT}/3rdparty/rknpu2/include
    ${PPOCR_VENDOR_ROOT}/3rdparty/stb_image
)
target_link_libraries(rk3568_ppocr_vendor PUBLIC
    ${OpenCV_LIBS}
    ${RKNN_LIBRARY}
    ${RGA_LIBRARY}
    ${TURBOJPEG_LIBRARY}
    Threads::Threads
    dl
    m
)

add_library(rk3568_ppocr STATIC src/ppocr_engine.cc)
target_include_directories(rk3568_ppocr PUBLIC include)
target_link_libraries(rk3568_ppocr PUBLIC rk3568_ppocr_vendor)

add_library(rk3568_docaligner STATIC
    src/docaligner_rknn.cc
)
target_include_directories(rk3568_docaligner
    PUBLIC include
    PRIVATE ${RKNN_INCLUDE_DIR}
)
target_link_libraries(rk3568_docaligner
    PUBLIC ${OpenCV_LIBS}
    PRIVATE ${RKNN_LIBRARY}
)

add_library(rk3568_camera_io STATIC
    src/rga_preprocessor.cc
    src/v4l2_capture.cc
)

add_library(rk3568_pipeline_core STATIC
    src/document_transform.cc
    src/frame_quality.cc
    src/npu_frequency_guard.cc
    src/paper_stability.cc
    src/structured_feedback.cc
)
target_include_directories(rk3568_pipeline_core PUBLIC include)
target_link_libraries(rk3568_pipeline_core PUBLIC ${OpenCV_LIBS})
target_include_directories(rk3568_camera_io
    PUBLIC include
    PRIVATE ${RGA_INCLUDE_DIR}
)
target_link_libraries(rk3568_camera_io
    PUBLIC ${OpenCV_LIBS}
    PRIVATE ${RGA_LIBRARY}
)

add_executable(rk3568_docaligner_probe
    src/docaligner_probe.cc
)
target_link_libraries(rk3568_docaligner_probe PRIVATE rk3568_docaligner)

add_executable(rk3568_camera_probe
    src/camera_probe.cc
)
target_link_libraries(rk3568_camera_probe
    PRIVATE rk3568_camera_io rk3568_docaligner rk3568_pipeline_core)

enable_testing()
add_executable(rk3568_pipeline_core_tests
    tests/document_transform_test.cc
    tests/paper_stability_test.cc
)

add_executable(rk3568_pipeline_probe src/pipeline_probe.cc)
target_link_libraries(rk3568_pipeline_probe PRIVATE
    rk3568_docaligner rk3568_pipeline_core rk3568_ppocr)
target_link_libraries(rk3568_pipeline_core_tests PRIVATE rk3568_pipeline_core)
target_compile_options(rk3568_pipeline_core_tests PRIVATE -UNDEBUG)
add_test(NAME pipeline_core COMMAND rk3568_pipeline_core_tests)

add_executable(rk3568_native_pipeline_service src/native_pipeline_service.cc)
target_link_libraries(rk3568_native_pipeline_service PRIVATE
    rk3568_camera_io rk3568_docaligner rk3568_pipeline_core rk3568_ppocr)

add_executable(rk3568_rga_crop_probe src/rga_crop_probe.cc)
target_link_libraries(rk3568_rga_crop_probe PRIVATE rk3568_camera_io)

install(TARGETS rk3568_docaligner_probe rk3568_camera_probe rk3568_pipeline_probe
                rk3568_native_pipeline_service
        RUNTIME DESTINATION bin)
