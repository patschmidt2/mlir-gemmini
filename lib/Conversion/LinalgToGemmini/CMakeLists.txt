add_mlir_conversion_library(LinalgToGemmini
  LinalgToGemmini.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/Conversion

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
