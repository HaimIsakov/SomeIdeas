/*
 * AttractionBasinWrapper.h
 *
 *  Created on: Jan 15, 2019
 *
 */

#ifndef SRC_WRAPPERS_FLOWWRAPPER_H_
#define SRC_WRAPPERS_FLOWWRAPPER_H_


#include "WrapperIncludes.h"
#include "../includes/FlowCalculator.h"
void BoostDefFlowCalculator();
py::list FlowCalculatorWrapper(dict converted_graph,double threshold=0);


#endif /* SRC_WRAPPERS_ATTRACTIONBASINWRAPPER_H_ */
