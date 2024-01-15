//===- GemminiOps.td - Standalone dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "GemminiDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Gemmini_FooOp : Gemmini_Op<"foo", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `standalone.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = constant 2 : i32
        // Apply the foo operation to %0
        %1 = standalone.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def Gemmini_FlushOp : Gemmini_Op<"flush">{
    let summary = "Flush operation";
    let description = [{Call the flush operation to flush the TLB}];

    let arguments = (ins I32:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";

}

def Gemmini_TiledMatmulAuto : Gemmini_Op<"tiled_matmul_auto">{
    let summary = "Calls an automatically tiled matmul function for Gemmini";
    let description = [{Performs a tiled matmul operation, e.g. on a fully connected Layer}];

    let arguments = (ins 
        MemRefRankOf<[AnyType], [2]>:$A,
        MemRefRankOf<[AnyType], [2]>:$B,
        MemRefRankOf<[AnyType], [2]>:$C,
        I32Attr:$stride_A,
        I32Attr:$stride_B,
        I32Attr:$stride_D,
        I32Attr:$stride_C,
        DefaultValuedAttr<F32Attr, "1.0">:$A_scale_factor,
        DefaultValuedAttr<F32Attr, "1.0">:$B_scale_factor,
        DefaultValuedAttr<F32Attr, "1.0">:$D_scale_factor,
        DefaultValuedAttr<I32Attr, "0">:$act,
        DefaultValuedAttr<F32Attr, "1.0">:$acc_scale,
        DefaultValuedAttr<F32Attr, "0.0">:$bert_scale,
        DefaultValuedAttr<BoolAttr, "true">:$repeating_bias,
        DefaultValuedAttr<BoolAttr, "false">:$transpose_A,
        DefaultValuedAttr<BoolAttr, "false">:$full_C,
        DefaultValuedAttr<BoolAttr, "false">:$low_D,
        DefaultValuedAttr<I8Attr, "0">:$weightA,
        DefaultValuedAttr<I32Attr, "1">:$tiled_matmul_type
    );

    let results = (outs 
        MemRefRankOf<[AnyType], [2]>:$D
    );

    let assemblyFormat = [{$stride_A $stride_B $stride_D $stride_C 
                        $A_scale_factor $B_scale_factor $D_scale_factor
                        $act
                        $acc_scale $bert_scale
                        $A $B $C 
                        attr-dict`:` type($A) type($B) type($C) type($D)}];
}

#endif // STANDALONE_OPS
