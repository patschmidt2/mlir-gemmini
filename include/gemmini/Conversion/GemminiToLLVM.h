#ifndef CONVERSION_GEMMINI_TO_LLVM_H_
#define CONVERSION_GEMMINI_TO_LLVM_H_

#include "mlir/Pass/Pass.h"


namespace mlir {
namespace gemmini {

#define GEN_PASS_DECL_GEMMINITOLLVM
#include "gemmini/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createGemminiToLLVM();


} //namespace gemmini
} //namespace mlir


#endif
