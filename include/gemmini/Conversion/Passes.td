#ifndef LIB_TRANSFORM_LINALG_TO_GEMMINI_
#define LIB_TRANSFORM_LINALG_TO_GEMMINI_

include "mlir/Pass/PassBase.td"

def LinalgToGemmini : Pass<"linalg-to-gemmini"> {
    let summary = "Lower 'linalg' operations to gemmini";

    let description = [{
        "Lower individual linalg.matmul and linalg.conv2d operations to the corresponding operation in the Gemmini dialect"
    }];

    let dependentDialects = [
        "mlir::gemmini::GemminiDialect",
        "mlir::func::FuncDialect",
        "mlir::memref::MemRefDialect"
    ];
}

#endif // LIB_TRANSFORM_LINALG_TO_GEMMINI_