// RUN: %PYTHON %S/Inputs/check_probability_packing.py triton-opt %s
// The driver checks HD64/HD128 fixtures at U2/U4, default identity, inert
// generic mode, unsupported-geometry rollback and invalid-option rejection.
module {}
