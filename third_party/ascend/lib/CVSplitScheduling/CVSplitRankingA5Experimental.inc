// Provisional A5 ranking hypotheses for Stage 8.5.
//
// The pressure rate approximates storage residency that is absent from the
// frozen principal-resource enum. It is structural and never keyed by
// candidate identity, kernel name, head dimension, or unroll factor.
constexpr uint64_t kExperimentalLiveResultPressureBytesPerCycle = 256;
constexpr uint64_t kExperimentalExposedWaitNormalization = 2;
constexpr uint32_t kExperimentalMinimumRankingMarginBasisPoints = 200;
