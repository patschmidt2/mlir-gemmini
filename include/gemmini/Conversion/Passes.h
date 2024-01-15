#ifndef GEMMINI_CONVERSION_PASSES_H_
#define GEMMINI_CONVERSION_PASSES_H_

#include "gemmini/Conversion/LinalgToGemmini/LinalgToGemmini.h"

namespace mlir {
namespace gemmini {

#define GEN_PASS_REGISTRATION
#include "gemmini/Conversion/Passes.h.inc"

} //namespace gemmini
} //namespace mlir


#endif //GEMMINI_CONVERSION_PASSES_H_