func.func @test_lower_matmul() -> (){
    %input = memref.alloc() : memref<8x8xi8>
    %weight = memref.alloc() : memref<8x8xi8>
    %result = memref.alloc() : memref<8x8xi8>

    linalg.matmul ins(%input, %weight : memref<8x8xi8>, memref<8x8xi8>)
                        outs(%result : memref<8x8xi8>)
        
    return
}