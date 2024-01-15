#ifndef LIB_TRANSFORM_LINALG_TO_GEMMINI_
#define LIB_TRANSFORM_LINALG_TO_GEMMINI_

include "mlir/Pass/PassBase.td"

def LinalgToGemmini : Pass<"linalg-to-gemmini"> {
    let summary = "Lower 'linalgg', operations to gemmini";

    let description = [{
        "TODO"
    }];

    let dependentDialects = [
        "mlir::func::FuncDialect",
        "mlir::memref::MemRefDialect"
    ];
}

#endif // LIB_TRANSFORM_LINALG_TO_GEMMINI_