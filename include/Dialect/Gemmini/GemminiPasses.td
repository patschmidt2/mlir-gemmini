#ifndef DIALECT_GEMMINI_PASSES_TD
#define DIALECT_GEMMINI_PASSES_TD

include "mlir/Pass/PassBase.td"

def LayerToTile : Pass<"gemmini-layer-to-tile"> {
    let summary = "Lower a layer operation to one that is tiled";

    let description = [{
        Given  a Gemmini Layer Operation, i.e. TiledMatmulAuto or TiledConv2dAuto, this
        pass creates splits the operation into tiles that fit into the on-chip memories of Gemmini.
    }];

    let dependentDialects = [
        "mlir::gemmini::GemminiDialect",
        "mlir::affine::AffineDialect",
        "mlir::scf::SCFDialect"
    ];

}


def TileToIntrin : Pass<"gemmini-tile-to-intrin">{
    let summary = "Lower from tile down to intrinsics";

    let description = [{
        Pass to lower from a tile operation down to a structure that calls Gemmini intrinsics.
    }];

    let dependentDialects = [
        "mlir::gemmini::GemminiDialect",
        "mlir::affine::AffineDialect",
        "mlir::scf::SCFDialect"
    ];
}


#endif // DIALECT_GEMMINI_PASSES_TD