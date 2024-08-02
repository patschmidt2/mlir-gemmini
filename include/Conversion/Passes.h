#ifndef GEMMINI_CONVERSION_PASSES_H_
#define GEMMINI_CONVERSION_PASSES_H_

#include "Conversion/LinalgToGemmini.h"
#include "Conversion/GemminiToLLVM.h"

namespace mlir {
namespace gemmini {

#define GEN_PASS_REGISTRATION
#include "Conversion/Passes.h.inc"

} //namespace gemmini
} //namespace mlir


#endif //GEMMINI_CONVERSION_PASSES_H_