// RUN: %PYTHON %S/Inputs/check_explicit_l0c.py triton-opt %S/cv_split_l0c_drain_control.mlir
// The existing attention fixtures provide the SSA input. This test checks
// compiler-generated explicit CC storage without depending on SSA spelling.
