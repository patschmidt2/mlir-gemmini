add_mlir_dialect_library(MLIRGemmini
        GemminiDialect.cpp
        GemminiOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Gemmini

        DEPENDS
        MLIRGemminiIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRInferTypeOpInterface
	)
