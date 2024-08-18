add_mlir_dialect_library(GemminiTransforms
    LayerToTile.cpp

    DEPENDS
    GemminiTransformIncGen

    LINK_LIBS PUBLIC
	MLIRIR
    MLIRInferTypeOpInterface
    MLIRSCFDialect
    MLIRAffineDialect

)