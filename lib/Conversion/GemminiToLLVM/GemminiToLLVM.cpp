#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include "Conversion/GemminiToLLVM.h"
#include "Gemmini/Dialect/GemminiDialect.h"
#include "Gemmini/Dialect/GemminiOps.h"

#define  GEN_PASS_DEF_GEMMINITOLLVM
#include "Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir {
namespace gemmini {

struct GemminiToLLVM : ::impl::GemminiToLLVMBase<GemminiToLLVM>{
    using GemminiToLLVMBase::GemminiToLLVMBase;

    void runOnOperation() override{
        MLIRContext *context = &getContext();
        auto *module = getOperation();

        ConversionTarget target(*context);
        target.addIllegalDialect<GemminiDialect>();

        RewritePatternSet patterns(context);

        if (failed(applyPartialConversion(module, target, std::move(patterns)))){
            signalPassFailure();
        }
    }
};

}
}