#include "Dialect/Gemmini/GemminiDialect.h"
#include "Dialect/Gemmini/GemminiOps.h"
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


namespace mlir{
namespace gemmini {

#define GEN_PASS_DEF_LAYERTOTILE
#include "Dialect/Gemmini/GemminiPasses.h.inc"

struct ConvertMatmulLayer : public OpConversionPattern<gemmini::TiledMatmulAuto>{
    using OpConversionPattern<gemmini::TiledMatmulAuto>::OpConversionPattern;

    ConvertMatmulLayer(MLIRContext *context)
        : OpConversionPattern<gemmini::TiledMatmulAuto>(context, 1) {}

    LogicalResult
    matchAndRewrite(gemmini::TiledMatmulAuto op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const final{


        return success();
    }
    
};

struct LayerToTile : impl::LayerToTileBase<LayerToTile>{
    using LayerToTileBase::LayerToTileBase;

    void runOnOperation() override {
        ConversionTarget target(getContext());

        target.addLegalDialect<BuiltinDialect,
                                affine::AffineDialect,
                                gemmini::GemminiDialect>();

        RewritePatternSet patterns(&getContext());
        //patterns.add<
    }

};



} // namespace gemmini
} // namespace mlir