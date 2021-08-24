/*
 * ExampleWrapper.h
 *
 *  Created on: Nov 11, 2018
 *
 */

#ifndef WRAPPERS_GPU_MOTIF_WRAPPER_H_
#define WRAPPERS_GPU_MOTIF_WRAPPER_H_

#include "../wrappers/WrapperIncludes.h"
#include "../includes/GPUMotifCalculator.h"

void BoostDefGPUMotifCalculator();
py::list GPUMotifCalculatorWrapper(dict converted_graph,int level, int cudaDevice);


#endif /* WRAPPERS_EXAMPLEWRAPPER_H_ */
