add_mlir_conversion_library(LinalgToGemmini
  GemminiToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/Conversion/

  DEPENDS
  GemminiConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRIR
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  MLIRTransforms
  )
