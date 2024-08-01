#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

#include "gemmini/Conversion/LinalgToGemmini.h"

#include "gemmini/Dialect/GemminiDialect.h"
#include "gemmini/Dialect/GemminiOps.h"

#define GEN_PASS_DEF_LINALGTOGEMMINI
#include "gemmini/Conversion/Passes.h.inc"

using namespace mlir;

namespace mlir{
namespace gemmini{

struct ConvertMatmul : public OpConversionPattern<linalg::MatmulOp> {
    using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

    ConvertMatmul(MLIRContext *context)
        : OpConversionPattern<linalg::MatmulOp>(context, 1) {}

    LogicalResult
    matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const final {
        auto loc = op.getLoc();

        auto ins =  op.getInputs();
        auto outs = op.getOutputs();

        // linalg.matmul calculates A*B = C
        auto matA = ins[0];
        auto matB = ins[1];
        auto matC = outs[0];

        MemRefType matAType = dyn_cast<MemRefType>(matA.getType());
        MemRefType matBType = dyn_cast<MemRefType>(matB.getType());
        MemRefType matCType = dyn_cast<MemRefType>(matC.getType());

        auto matAShape = matAType.getShape();
        auto matBShape = matBType.getShape();
        auto matCShape = matCType.getShape();

        // Create bias
        MemRefType biasType = MemRefType::get({matCShape}, rewriter.getI32Type());
        Value bias = rewriter.create<memref::AllocOp>(loc, biasType);

        Type fillOpInType = rewriter.getI32Type();
        TypedAttr fillOpInputAttr = rewriter.getI32IntegerAttr(0);
        Value fillOpValue = rewriter.create<arith::ConstantOp>(loc, fillOpInType, fillOpInputAttr);
        rewriter.create<linalg::FillOp>(loc, fillOpValue, bias); //fill bias with zeros

        //int dimI = matAShape[0];
        int dimJ = matAShape[1];
        int dimK = matBShape[1];

        llvm::APFloat scale1((float)1.0);
        llvm::APFloat scale0((float)0.0);

        rewriter.replaceOpWithNewOp<gemmini::TiledMatmulAuto>(op, 
                    matA, matB, matC, bias,
                    dimK, dimJ, dimJ, dimJ);
        
        rewriter.create<memref::DeallocOp>(loc, bias);

        return success();

    }

};
    

struct ConvertConv2d : public OpConversionPattern<linalg::Conv2DNhwcHwcfOp>{
    using OpConversionPattern<linalg::Conv2DNhwcHwcfOp>::OpConversionPattern;

    ConvertConv2d(MLIRContext *context)
        : OpConversionPattern<linalg::Conv2DNhwcHwcfOp>(context, 1) {}

    LogicalResult
    matchAndRewrite(linalg::Conv2DNhwcHwcfOp op, OpAdaptor adaptor,
                ConversionPatternRewriter &rewriter) const final {
        Location loc = op.getLoc();

        auto ins =  op.getInputs();
        auto outs = op.getOutputs();

        auto input = ins[0];
        auto weight = ins[0];
        auto result = outs[0];

        //MemRefType inputType = dyn_cast<MemRefType>(input.getType());
        //MemRefType weightType = dyn_cast<MemRefType>(weight.getType());
        MemRefType resultType = dyn_cast<MemRefType>(result.getType());

        // Check if this op can be converted to Gemmini
        if(!llvm::all_equal(op.getStrides())){
            return failure();
        }
        auto apStride = *(op.getStrides().begin());
        uint32_t stride = apStride.getZExtValue();

        // Create bias
        MemRefType biasType = MemRefType::get({resultType.getShape()[3]}, rewriter.getI32Type());
        Value bias = rewriter.create<memref::AllocOp>(loc, biasType);

        Type fillOpInType = rewriter.getI32Type();
        TypedAttr fillOpInputAttr = rewriter.getI32IntegerAttr(0);
        Value fillOpValue = rewriter.create<arith::ConstantOp>(loc, fillOpInType, fillOpInputAttr);
        rewriter.create<linalg::FillOp>(loc, fillOpValue, bias); //fill bias with zeros

        // Create constants
        //llvm::APFloat scale1(float(1.0));
        //uint32_t int1 = 1;
        //uint32_t int0 = 0;

        rewriter.replaceOpWithNewOp<gemmini::TiledConvAuto>(op,
                                                            resultType,
                                                            input, weight, result, bias, 
                                                            stride
                                                            );
        //rewriter.replaceOpWithNewOp<gemmini::TiledConvAuto>(op,
        //                                                    resultType,
        //                                                    input, weight, result, bias, 
        //                                                    stride, 
        //                                                    int1, int1, int0,
        //                                                    false, false, false, false, false,
        //                                                    int0, scale1, 
        //                                                    int1, int0, int0);

        rewriter.create<memref::DeallocOp>(loc, bias);

        return success();
    }
};

//===----------------------------------------------------------------------===//
// LinalgToGemmini Lowering Pass 
//===----------------------------------------------------------------------===//

struct LinalgToGemmini :  ::impl::LinalgToGemminiBase<LinalgToGemmini> {
    void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<BuiltinDialect, 
                        affine::AffineDialect, 
                        arith::ArithDialect, 
                        memref::MemRefDialect, 
                        linalg::LinalgDialect,
                        gemmini::GemminiDialect>();
    target.addIllegalOp<linalg::MatmulOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertMatmul>(
        &getContext());


    if(failed(
            applyPartialConversion(getOperation(), target, std::move(patterns)))){
        signalPassFailure();
    }
    }
};

std::unique_ptr<Pass> createLinalgToGemmini(){
    return std::make_unique<LinalgToGemmini>();
}

/// This is a partial lowering from linalg operations to the gemmini dialect
//struct LinalgToGemminiLoweringPass : public PassWrapper<LinalgToGemminiLoweringPass, OperationPass<ModuleOp>> {
//    //MLIR_DEFINE_EXPLICIT_INTERNAL_TYPE_ID(LinalgToGemminiLoweringPass)
//
//    void getDependentDialects(DialectRegistry &registry) const override{
//        registry.insert<func::FuncDialect, memref::MemRefDialect>();
//    }
//
//    void runOnOperation() final;
//};
//
//
//
//void LinalgToGemminiLoweringPass::runOnOperation() {
//    ConversionTarget target(getContext());
//
//    target.addLegalDialect<BuiltinDialect, 
//                        affine::AffineDialect, 
//                        arith::ArithDialect, 
//                        memref::MemRefDialect, 
//                        linalg::LinalgDialect,
//                        gemmini::GemminiDialect>();
//    target.addIllegalOp<linalg::MatmulOp>();
//
//    RewritePatternSet patterns(&getContext());
//    patterns.add<ConvertMatmul>(
//        &getContext());
//
//
//    if(failed(
//            applyPartialConversion(getOperation(), target, std::move(patterns)))){
//        signalPassFailure();
//    }
//}
//
//std::unique_ptr<Pass> createLowerLinalgToGemminiPass(){
//    return std::make_unique<LinalgToGemminiLoweringPass>();
//}

} //namespace gemmini
} //namespace mlir