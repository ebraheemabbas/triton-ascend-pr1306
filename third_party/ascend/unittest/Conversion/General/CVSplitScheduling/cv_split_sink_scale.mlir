// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 sink-scale-into-fixpipe=true" 2>/dev/null | FileCheck %s --check-prefix=SINK
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=KEEP
//
// The softmax scale is one multiply of the score tile by a constant splat,
// and the fixpipe that delivers the score tile can apply a scalar during the
// move it already makes.  When the tile's only VECTOR consumer is that
// multiply, the scalar rides the fixpipe as quant_scale -- with QF322F32_PRE
// spelled out, because the f32->f32 lowering derives no pre-quant mode on its
// own and would hand the scalar to a library call that ignores it -- and the
// multiply, its splat and the splat's template all disappear.
//
// Only the score role qualifies.  The product tile's consumer is the
// accumulator update, not a scale, so the four PV fixpipes stay bare; the
// count pins the absorbed multiplies to exactly the four QK lanes.
//
// The transform is off by default and this test drives it explicitly: on the
// FA kernel it measured 10.7% slower at hd=64, because the fixpipe time it
// adds (305us) exceeds the vector time it saves (187us) and the fixpipe is
// the contended pipe.  KEEP therefore covers the default path.

// The splat template is never materialized...
// SINK-NOT: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<16x32xf32>)
// ...and exactly the four score fixpipes carry the scalar.
// SINK-COUNT-4: hivm.hir.fixpipe {{.*}} quant_scale = %{{.*}} : f32 dual_dst_mode = {{<}}ROW_SPLIT{{>}}
// SINK-NOT: quant_scale
// No score-shaped multiply survives on the VECTOR side either.
// SINK-NOT: arith.mulf {{.*}} : tensor<16x32xf32>

// The default keeps the multiply path: the splat is materialized, no
// fixpipe carries a scalar, and the VECTOR scope keeps the multiply.
// KEEP: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<16x32xf32>)
// KEEP-NOT: quant_scale
// KEEP-NOT: pre_quant
// KEEP: arith.mulf {{.*}} : tensor<16x32xf32>
