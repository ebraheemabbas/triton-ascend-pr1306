// RUN: %PYTHON %S/Inputs/check_policy_defaults.py triton-opt %S/cv_split_asymmetric_materializer.mlir

// The helper checks HD64 and HD128 using the existing full-size fixtures.
// Generic-path tests set their legacy policy explicitly; this test exercises
// actual pass defaults and the unchanged candidate=-1 rollback contract.
