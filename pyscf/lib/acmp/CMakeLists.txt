add_library(acmp SHARED mp2.c)

set_target_properties(acmp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
  COMPILE_FLAGS ${OpenMP_C_FLAGS}
  LINK_FLAGS ${OpenMP_C_FLAGS})

target_link_libraries(acmp cvhf ${BLAS_LIBRARIES})

