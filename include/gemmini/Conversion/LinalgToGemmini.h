#ifndef CONVERSION_LINALG_TO_GEMMINI_H_
#define CONVERSION_LINALG_TO_GEMMINI_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace gemmini {

#define GEN_PASS_DECL_LINALGTOGEMMINI
#include "gemmini/Conversion/Passes.h.inc"

std::unique_ptr<mlir::Pass> createLinalgToGemmini();

} //namespace gemmini
} //namespace mlir


#endif