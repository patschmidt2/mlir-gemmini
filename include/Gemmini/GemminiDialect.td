//===- StandaloneDialect.td - Standalone dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GEMMINI_DIALECT
#define GEMMINI_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def Gemmini_Dialect : Dialect {
    let name = "gemmini";
    let summary = "A gemmini out-of-tree MLIR dialect.";
    let description = [{
        This is a dialect to import models via mlir-torch and lower them to gemmini operations
    }];
    let cppNamespace = "::mlir::gemmini";
}

//===----------------------------------------------------------------------===//
// Base gemmini operation definition.
//===----------------------------------------------------------------------===//

class Gemmini_Op<string mnemonic, list<Trait> traits = []> :
        Op<Gemmini_Dialect, mnemonic, traits>;

#endif // STANDALONE_DIALECT
