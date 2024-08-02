//===- standalone-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "gemmini/Dialect/GemminiDialect.h"
#include "gemmini/Dialect/GemminiDialect.cpp.inc"

#include "gemmini/Conversion/Passes.h"

int main(int argc, char **argv) {
  // TODO: Register standalone passes here.

  mlir::DialectRegistry registry;
  registry.insert<mlir::gemmini::GemminiDialect>();

  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();

  mlir::gemmini::registerLinalgToGemmini();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Standalone optimizer driver\n", registry));
}
