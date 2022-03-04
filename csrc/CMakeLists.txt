cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(DeepPool)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/nfsnccl")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/libtorch")
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/cuda")
find_package(Torch REQUIRED)
message("TORCH_LIBRARIES:${TORCH_LIBRARIES}")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -std=c++17 -lstdc++fs -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

enable_language(CUDA)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
set(BUILD_SHARED_LIBS ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "80")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -std=c++17)

cuda_add_library("cudasleep"
  CUDASleep.cu
  SHARED
)

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

add_subdirectory(grpc)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
  find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
  set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# Proto file
get_filename_component(dp_proto "protos/runtime.proto" ABSOLUTE)
get_filename_component(dp_proto_path "${dp_proto}" PATH)

# Generated sources
set(dp_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/runtime.pb.cc")
set(dp_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/runtime.pb.h")
set(dp_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/runtime.grpc.pb.cc")
set(dp_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/runtime.grpc.pb.h")
add_custom_command(
      OUTPUT "${dp_proto_srcs}" "${dp_proto_hdrs}" "${dp_grpc_srcs}" "${dp_grpc_hdrs}"
      COMMAND ${_PROTOBUF_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${dp_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${dp_proto}"
      DEPENDS "${dp_proto}")


if(${BUILD_ANVIL})
    include(FetchContent)
    FetchContent_Declare(matplotplusplus
            GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
            GIT_TAG 3fd7593f3bf98ca60fdf3db17073ea8bbe161263 # or whatever tag you want
    )

    FetchContent_GetProperties(matplotplusplus)
    if(NOT matplotplusplus_POPULATED)
        FetchContent_Populate(matplotplusplus)
        add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()

    message("matplotplusplus_SOURCE_DIR:${matplotplusplus_SOURCE_DIR}")
    message("matplotplusplus_BINARY_DIR:${matplotplusplus_BINARY_DIR}")
endif()

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

include_directories("$ENV{HOME}/nfsnccl/include")
link_directories("$ENV{HOME}/nfsnccl/lib")

include_directories("$ENV{HOME}/nfsnccl/include")
link_directories("$ENV{HOME}/nfsnccl/lib")

# dp_grpc_proto
add_library(dp_grpc_proto
  ${dp_grpc_srcs}
  ${dp_grpc_hdrs}
  ${dp_proto_srcs}
  ${dp_proto_hdrs})

target_link_libraries(dp_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})


if(${BUILD_ANVIL})
    find_package(HDF5 REQUIRED)
    include_directories(${HDF5_INCLUDE_DIRS})

    find_package(OpenCV REQUIRED)
    message(STATUS "OpenCV library status:")
    message(STATUS "    config: ${OpenCV_DIR}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
    
  add_executable(runtime
    "runtime.cpp" "runnableModule.cpp" "JobContext.cpp" "logger.cpp"
    "communication.cpp" "rpcService.cpp" "GraphPieces.cpp" "BeTask.cpp" 
    "CodeLocation.cpp" "Cycles.cpp" "SpinLock.cpp" "ThreadId.cpp" "cifar10.cpp" 
    "dataset.cpp" "GradSync.cpp" "CUDAGraph.cpp"
    "streamingDataset.cpp" "stats.cpp"
  )
  target_link_libraries(runtime
    dp_grpc_proto
    cudasleep
    absl::flags absl::flags_parse
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
    ${HDF5_LIBRARIES}
    matplot
  )
else()
  add_executable(runtime
    "runtime.cpp" "runnableModule.cpp" "JobContext.cpp" "logger.cpp"
    "communication.cpp" "rpcService.cpp" "GraphPieces.cpp" "BeTask.cpp" 
    "CodeLocation.cpp" "Cycles.cpp" "SpinLock.cpp" "ThreadId.cpp" "cifar10.cpp" 
    "dataset.cpp" "GradSync.cpp" "CUDAGraph.cpp"
  )
  target_link_libraries(runtime
    dp_grpc_proto
    cudasleep
    absl::flags absl::flags_parse
    ${_REFLECTION}
    ${_GRPC_GRPCPP}
    ${_PROTOBUF_LIBPROTOBUF}
    ${TORCH_LIBRARIES}
  )
endif()
  
# opencv_imgcodecs
set_property(TARGET runtime PROPERTY CXX_STANDARD 17)
