// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 regroup-softmax-max=true" 2>/dev/null | FileCheck %s --check-prefix=DRAIN
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 regroup-softmax-max=true" 2>/dev/null | FileCheck %s --check-prefix=RESCALE
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4 regroup-softmax-max=true" 2>/dev/null | FileCheck %s --check-prefix=FLAGS
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=STREAMDRAIN
// RUN: triton-opt %S/cv_split_scheduling_fa.mlir "--cv_split_scheduling=compile-on-910-95=true unroll-factor=4" 2>/dev/null | FileCheck %s --check-prefix=STREAMFLAGS
//
// Streaming softmax takes a running maximum after every block, so each unrolled
// lane carries its own rescale factor, and each lane's product has to come back
// to VECTOR to be folded into the accumulator.  Taking one maximum over the
// whole group before any exponential puts the P tiles on a common scale: the
// later lanes' factors are exp(0) = 1 and disappear, the accumulator is
// rescaled once, and the products accumulate into each other so the chain stays
// on CUBE and only its result crosses back.
//
// Same arithmetic, different association -- the group maximum is at least every
// lane's, so every exponent stays <= 0.  It is still a bit-for-bit change, so
// the option is off by default and wants an accuracy run behind it.
//
// Counts of different operations need separate prefixes: they interleave in the
// output, and FileCheck matches a prefix's directives in order.

// Four score drains stay; the four product drains collapse into one.
// DRAIN-COUNT-5: hivm.hir.fixpipe
// DRAIN-NOT: hivm.hir.fixpipe

// Four score exponentials and one rescale factor, where the streaming form
// computes a factor per lane.
// RESCALE-COUNT-5: math.exp
// RESCALE-NOT: math.exp

// Four score hand-offs, four P hand-offs, one product hand-off and the
// back-edge release: seven flags, against thirteen for the streaming form.
// FLAGS: flag = 6
// FLAGS-NOT: flag = 7

// Without the option nothing changes: a drain and a rescale per lane.
// STREAMDRAIN-COUNT-8: hivm.hir.fixpipe
// STREAMDRAIN-NOT: hivm.hir.fixpipe
// STREAMFLAGS: flag = 12
