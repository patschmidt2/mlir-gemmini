// RUN: gemmini-opt %s | gemmini-opt | FileCheck %s

module {
    // CHECK-LABEL: func @bar()
    func.func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = gemmini.foo %{{.*}} : i32
        %res = gemmini.foo %0 : i32
        return
    }
}
