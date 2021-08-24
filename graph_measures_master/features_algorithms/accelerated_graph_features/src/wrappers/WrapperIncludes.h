/*
 * WrapperIncludes.h
 *
 *  Created on: Nov 12, 2018
 *
 */

#ifndef WRAPPERS_WRAPPERINCLUDES_H_
#define WRAPPERS_WRAPPERINCLUDES_H_

#include <boost/python.hpp>
#include <boost/python/dict.hpp>
#include "../includes/ConvertedGNXReciever.h"
#include "../includes/FeatureCalculator.h"

#include <boost/python/list.hpp>
#include <boost/python/tuple.hpp>
#include <vector>
#include <tuple>

typedef std::tuple<float,float> floatTuple;

using namespace boost::python;

namespace py = boost::python;


template<class T>
py::list vectorToPythonList(const std::vector<T>& v){
	py::list l;
	for(int i=0;i<v.size();i++){
		l.append<T>(v[i]);
	}

	return l;
}

py::list convertVectorOfVectorsTo2DList(vector<vector<unsigned int>*>* vec);






#endif /* WRAPPERS_WRAPPERINCLUDES_H_ */
