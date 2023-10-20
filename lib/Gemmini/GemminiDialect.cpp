//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Gemmini/GemminiDialect.h"
#include "Gemmini/GemminiOps.h"

using namespace mlir;
using namespace mlir::gemmini;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void GemminiDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Gemmini/GemminiOps.cpp.inc"
      >();
}
