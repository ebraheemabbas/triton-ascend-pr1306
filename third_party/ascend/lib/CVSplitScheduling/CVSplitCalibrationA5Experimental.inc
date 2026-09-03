// Provisional A5 calibration hypotheses for Stage 8.3.
//
// These constants exercise the frozen v4 API and deterministic scheduler
// integration. They are not qualified absolute hardware timings. Every change
// requires a new calibration ID and versioned evidence.
constexpr uint32_t kExperimentalA5ProductId = 9579;
constexpr uint32_t kExperimentalA5Revision = 0;
constexpr uint32_t kExperimentalCalibrationSchemaVersion = 3;
constexpr CVSplitCalibrationId kExperimentalCalibrationId{
    0xA500000000000805ULL};

constexpr uint32_t kMaximumCalibratedMatrixDimension = 4096;
constexpr uint64_t kMaximumCalibratedBytes = 1ULL << 30;

constexpr uint64_t kL1InputStartupCycles = 8;
constexpr uint64_t kL1InputBytesPerCycle = 512;
constexpr uint64_t kUbInputStartupCycles = 16;
constexpr uint64_t kUbInputBytesPerCycle = 384;
constexpr uint64_t kGlobalInputStartupCycles = 32;
constexpr uint64_t kGlobalInputBytesPerCycle = 256;
constexpr uint64_t kMatrixStartupCycles = 24;
constexpr uint64_t kMatrixMacsPerCycle = 4096;
constexpr uint64_t kCubeDrainStartupCycles = 12;
constexpr uint64_t kCubeDrainBytesPerCycle = 256;

constexpr uint64_t kVectorRegionStartupCycles = 12;
constexpr uint64_t kVectorFragmentDispatchCycles = 2;
constexpr uint64_t kVectorDependencyDepthCycles = 2;
constexpr uint64_t kVectorElementwiseElementsPerCycle = 256;
constexpr uint64_t kVectorMaximumElementsPerCycle = 128;
constexpr uint64_t kVectorExpElementsPerCycle = 32;
constexpr uint64_t kVectorReductionElementsPerCycle = 64;
constexpr uint64_t kVectorCastElementsPerCycle = 256;
constexpr uint64_t kVectorPermuteElementsPerCycle = 128;
constexpr uint64_t kVectorCopyElementsPerCycle = 128;
constexpr uint64_t kVectorUbBytesPerCycle = 512;
constexpr uint64_t kVectorParallelSubBlocks = 2;

constexpr uint64_t kTransferCopyStartupCycles = 8;
constexpr uint64_t kTransferConversionStartupCycles = 14;
constexpr uint64_t kTransferFixpipeStartupCycles = 16;
constexpr uint64_t kTransferMteBytesPerCycle = 256;
constexpr uint64_t kTransferFixpipeBytesPerCycle = 256;
constexpr uint64_t kTransferRowGroup = 32;
constexpr uint64_t kTransferFusedScaleCycles = 8;

constexpr uint64_t kEventSetCycles = 2;
constexpr uint64_t kEventWaitIssueCycles = 2;
constexpr uint64_t kOwnershipReleaseCycles = 3;
constexpr uint64_t kLoopSeedCycles = 2;
constexpr uint64_t kCrossCoreSyncCycles = 1;
constexpr uint64_t kLoopCarriedSyncCycles = 1;

constexpr uint32_t kCubeUncertaintyBasisPoints = 2500;
constexpr uint32_t kVectorUncertaintyBasisPoints = 3500;
constexpr uint32_t kTransferUncertaintyBasisPoints = 3000;
constexpr uint32_t kSynchronizationUncertaintyBasisPoints = 5000;
