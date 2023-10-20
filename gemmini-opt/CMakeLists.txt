get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRGemmini
        )
add_llvm_executable(gemmini-opt gemmini-opt.cpp)

llvm_update_compile_flags(gemmini-opt)
target_link_libraries(gemmini-opt PRIVATE ${LIBS})
