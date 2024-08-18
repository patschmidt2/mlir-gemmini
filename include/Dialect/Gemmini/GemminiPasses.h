#ifndef GEMMINI_GEMMINIPASSES_H
#define GEMMINI_GEMMINIPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>


namespace mlir {
namespace gemmini{

#define GEN_PASS_REGISTRATION
#include "Dialect/Gemmini/GemminiPasses.h.inc"

std::unique_ptr<mlir::Pass> createLayerToTile();

} // namespace gemmini
} // namespace mlir

#endif // GEMMINI_GEMMINIPASSES_H