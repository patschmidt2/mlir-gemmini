add_mlir_conversion_library(LinalgToGemmini
  LinalgToGemmini.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/gemmini/Conversion/LinalgToGemmini

  DEPENDS
  GemminiConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  )


#add_mlir_conversion_library(LinalgToGemmini
#  LinalgToGemmini.cpp
#
#  ADDITIONAL_HEADER_DIRS
#  ${CMAKE_SOURCE_DIR}/include/gemmini
#
#  LINK_LIBS PUBLIC
#  MLIRLinalgDialect
#  MLIRMemRefDialect
#  MLIRPass
#  MLIRTransforms
#  MLIRAffineDialect
#  )
#
#set(LLVM_TARGET_DEFINITIONS LinalgToGemmini.td)
#mlir_tablegen(LinalgToGemmini.h.inc -gen-pass-decls -name LinalgToGemmini)
#add_dependencies(mlir-headers MLIRGemminiOpsIncGen)
#add_public_tablegen_target(LinalgToGemminiPassIncGen)
#add_mlir_doc(LinalgToGemmini LinalgToGemmini LinalgToGemmini/ -gen-pass-doc)
