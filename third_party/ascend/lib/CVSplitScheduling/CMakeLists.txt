add_triton_library(CVSplitScheduling
  CVSplitScheduling.cpp
  CrossCorePipelinePlan.cpp
  CrossCoreResourcePlan.cpp
  CrossScopeTransfers.cpp
  classifyAllOps.cpp
  DependencyScheduler.cpp
  PreCheck.cpp
  ScopeSeparation.cpp
  SoftmaxRegroup.cpp
  UnfusePVMatmuls.cpp

  DEPENDS
  CVSplitSchedulingPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  BiShengIRAnnotationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
  DynamicCVPipeline
)
