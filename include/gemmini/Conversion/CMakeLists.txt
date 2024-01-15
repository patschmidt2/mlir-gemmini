#add_mlir_conversion_library(LinalgToGemmini
#  LinalgToGemmini.cpp
#
#  ADDITIONAL_HEADER_DIRS
#  ${PROJ_INCLUDE_DIR}/gemmini/Conversion/LinalgToGemmini
#
#  DEPENDS
#  GemminiConversionPassIncGen
#
#  LINK_LIBS PUBLIC
#  MLIRAffineDialect
#  MLIRAffineToStandard
#  MLIRIR
#  MLIRLinalgDialect
#  MLIRMemRefDialect
#  MLIRPass
#  MLIRFuncDialect
#  MLIRSupport
#  MLIRTransforms
#  )

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(GemminiConversionPassIncGen)

add_mlir_doc(Passes ConversionPasses ./ -gen-pass-doc)