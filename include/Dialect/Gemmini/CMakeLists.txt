add_mlir_dialect(Gemmini gemmini)

#set(LLVM_TARGET_DEFINITIONS GemminiPasses.td)
#mlir_tablegen(GemminiPasses.h.inc -gen-pass-decls)
#add_public_tablegen_target(GemminiTransformIncGen)
#add_mlir_doc(GemminiPasses GemminiPasses Gemmini/ -gen-pass-doc)

add_mlir_doc(GemminiDialect GemminiDialect Gemmini/ -gen-dialect-doc)
add_mlir_doc(GemminiOps GemminiOps Gemmini/ -gen-op-doc)
