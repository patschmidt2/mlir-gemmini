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
        MemRefRankOf<[AnyType], [2]>:$D,
        MemRefRankOf<[AnyType], [1, 2]>:$C,
        I32Attr:$stride_A,
        I32Attr:$stride_B,
        I32Attr:$stride_D,
        I32Attr:$stride_C
        //DefaultValuedAttr<F32Attr, "1.0">:$A_scale_factor,
        //DefaultValuedAttr<F32Attr, "1.0">:$B_scale_factor,
        //DefaultValuedAttr<F32Attr, "1.0">:$D_scale_factor,
        //DefaultValuedAttr<I32Attr, "0">:$act,
        //DefaultValuedAttr<F32Attr, "1.0">:$acc_scale,
        //DefaultValuedAttr<F32Attr, "0.0">:$bert_scale,
        //DefaultValuedAttr<BoolAttr, "true">:$repeating_bias,
        //DefaultValuedAttr<BoolAttr, "false">:$transpose_A,
        //DefaultValuedAttr<BoolAttr, "false">:$full_C,
        //DefaultValuedAttr<BoolAttr, "false">:$low_D,
        //DefaultValuedAttr<I8Attr, "0">:$weightA,
        //DefaultValuedAttr<I32Attr, "1">:$tiled_matmul_type
    );


    //let assemblyFormat = [{$stride_A $stride_B $stride_D $stride_C 
    //                    $A_scale_factor $B_scale_factor $D_scale_factor
    //                    $act
    //                    $acc_scale $bert_scale
    //                    $A $B $C 
    //                    attr-dict`:` type($A) type($B) type($C) type($D)}];
}

def Gemmini_TiledConvAuto : Gemmini_Op<"tiled_conv_auto">{
    let summary = "Call automatically tiled vection of convolution operation";
    let description = [{Compute a full convolution layer}];

    let arguments = (ins
        MemRefRankOf<[AnyType], [4]>:$input,
        MemRefRankOf<[AnyType], [4]>:$weight,
        MemRefRankOf<[AnyType], [4]>:$result,
        MemRefRankOf<[AnyType], [4]>:$bias,
        I32Attr:$stride
        //DefaultValuedAttr<I32Attr, "1">:$input_dilation,
        //DefaultValuedAttr<I32Attr, "1">:$kernel_dilation,
        //DefaultValuedAttr<I32Attr, "0">:$padding,
        //DefaultValuedAttr<BoolAttr, "false">:$wrot180,
        //DefaultValuedAttr<BoolAttr, "false">:$trans_output_1203,
        //DefaultValuedAttr<BoolAttr, "false">:$trans_input_3120,
        //DefaultValuedAttr<BoolAttr, "false">:$trans_weight_1203,
        //DefaultValuedAttr<BoolAttr, "false">:$trans_weight_0132,
        //DefaultValuedAttr<I32Attr, "0">:$act,
        //DefaultValuedAttr<F32Attr, "1.0">:$scale,
        //DefaultValuedAttr<I32Attr, "1">:$pool_size,
        //DefaultValuedAttr<I32Attr, "0">:$pool_stride,
        //DefaultValuedAttr<I32Attr, "0">:$pool_padding
    );
}

def Gemmini_ExtendedMvin1 : Gemmini_Op<"extended_mvin1">{
    let summary  = "Move in a block of data that is of size (rows, cols)";
    let description =  [{Uses mvin engine Nr. 1 of 3 available ones. Typically used for input data}];

    let arguments = (ins
        MemRefRankOf<[AnyType], [1]>:$dram_addr,
        MemRefRankOf<[AnyType], [1]>:$spad_addr,
        I32:$cols,
        I32:$rows
    );
}

def Gemmini_ExtendedMvin2 : Gemmini_Op<"extended_mvin2">{
    let summary  = "Move in a block of data that is of size (rows, cols)";
    let description =  [{Uses mvin engine Nr. 2 of 3 available ones. Typically used for weights}];

    let arguments = (ins
        MemRefRankOf<[AnyType], [1]>:$dram_addr,
        MemRefRankOf<[AnyType], [1]>:$spad_addr,
        I32:$cols,
        I32:$rows
    );
}

def Gemmini_ExtendedMvin3 : Gemmini_Op<"extended_mvin3">{
    let summary  = "Move in a block of data that is of size (rows, cols)";
    let description =  [{Uses mvin engine Nr. 3 of 3 available ones. Typically used for weights}];

    let arguments = (ins
        MemRefRankOf<[AnyType], [1]>:$dram_addr,
        MemRefRankOf<[AnyType], [1]>:$spad_addr,
        I32:$cols,
        I32:$rows
    );
}

def Gemmini_ExtendedMvout : Gemmini_Op<"extended_mvout">{
    let summary  = "Move out a block of data that is of size (rows, cols)";
    let description =  [{Uses the available mvout engine of the DMA}];

    let arguments = (ins
        MemRefRankOf<[AnyType], [1]>:$dram_addr,
        MemRefRankOf<[AnyType], [1]>:$spad_addr,
        I32:$cols,
        I32:$rows
    );
}

def Gemmini_ExtendedComputePreloaded : Gemmini_Op<"extended_compute_preloaded">{
    let summary = "Perform a computation";
    let description = [{"Computation uses data preloaded to the array"}];

    let arguments = (ins
        MemRefRankOf<[AnyType], [1]>:$A_addr,
        MemRefRankOf<[AnyType], [1]>:$BD_addr,
        I32:$A_cols,
        I32:$A_rows,
        I32:$BD_cols,
        I32:$BD_rows
    );
}

def Gemmini_ExtendedComputeAccumulated : Gemmini_Op<"extended_compute_accumulated">{
    let summary = "Perform a computation";
    let description = [{"Computation accumulates on previously preloaded data"}];

    let arguments = (ins
        MemRefRankOf<[AnyType], [1]>:$A_addr,
        MemRefRankOf<[AnyType], [1]>:$BD_addr,
        I32:$A_cols,
        I32:$A_rows,
        I32:$BD_cols,
        I32:$BD_rows
    );
}

def Gemmini_ExtendedPreload : Gemmini_Op<"extended_preload">{
    let summary = "Perform a preload";
    let description = [{Loads data from the SPAD or ACC into the systolic array. Results of the computation will be written to C_addr}];

    let arguments = (ins
        MemRefRankOf<[AnyType], [1]>:$BD_addr,
        MemRefRankOf<[AnyType], [1]>:$C_addr,
        I32:$BD_cols,
        I32:$BD_rows,
        I32:$C_cols,
        I32:$C_rows
    );
}

def Gemmini_ExtendedConfigEx : Gemmini_Op<"extended_config_ex">{
    let summary = "Config the execution of matmul ops";
    let description = [{"Configuratoin parameters:
        dataflow: OS=1 or WS=1
        sys_act: Activation function to apply directly after computation, before storing to accumulator
        sys_shift: Shift to apply during computation
        sys_acc_scale: Scale to apply before (?) move to accumulator
        C_stride: Stride with which accesses to output are performed
        A_stride: Stride with which accesses to input data are performed
        A_transpose: Transpose A or not
        B_transpose: Transpose B or not
        set_only_strides: ignore all parameters besides stride
    "}];

    let arguments =  (ins
        I32:$dataflow,
        I32:$sys_act,
        I32:$sys_shift,
        I32:$sys_acc_scale,
        I32:$C_stride,
        I32:$A_stride,
        BoolAttr:$A_transpose,
        BoolAttr:$B_transpose,
        BoolAttr:$set_only_strides
    );
}

def Gemmini_ExtendedConfigLd : Gemmini_Op<"extended_config_ld">{
    let summary = "Configure execution of the data loads";
    let description = [{Configure data load operations}];

    let arguments = (ins
        I32:$stride,
        I32:$scale,
        BoolAttr:$shrunk,
        I32:$block_mvin_stride,
        I32:$pixel_repeats,
        I32:$id
    );
}

def Gemmini_ExtendedConfigSt : Gemmini_Op<"extended_config_st">{
    let summary = "Configure execution of the data stores";
    let description = [{Configure data store operations}];

    let arguments = (ins
        I32:$stride,
        I32:$acc_act,
        I32:$acc_scale,
        I32:$pool_stride,
        I32:$pool_size,
        I32:$pool_out_dim,
        I32:$porows,
        I32:$pocols,
        I32:$orows,
        I32:$ocols,
        I32:$upad,
        I32:$lpad
    );
}


#endif // STANDALONE_OPS
