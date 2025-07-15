#include <odepack/pyode.hpp>


using Tt = double;
using Tf = vec<double, -1>;

const int N = -1;

void hhode(Tf& res, const Tt& t, const Tf& q, const std::vector<Tt>& args){
    res[0] = q[2];
    res[1] = q[3];
    res[2] = -(std::pow(args[4], 2.)*q[0] + args[0]*(std::pow(q[1], 2.) + 3.*args[1]*std::pow(q[0], 2.) + 2.*args[2]*q[0]*q[1]));
    res[3] = -(std::pow(args[5], 2.)*q[1] + args[0]*(2.*q[0]*q[1] + args[2]*std::pow(q[0], 2.) + 3.*args[3]*std::pow(q[1], 2.)));
}


Tt event(const Tt& t, const Tf& q, const std::vector<Tt>& args){
    return q[1];
}

bool check_if(const Tt& t, const Tf& q, const std::vector<Tt>& args){
    return q[3] > 0;
}

PreciseEvent<Tt, N> ps_event("Poincare Section", event, check_if, nullptr, false, 1e-15);
std::vector<Event<Tt, N>*> events = {&ps_event};

const void* ptr1 = reinterpret_cast<const void*>(hhode);
const void* ptr2 = reinterpret_cast<const void*>(&events);

PYBIND11_MODULE(henon, m){

    m.def("_ptrs", [](){ py::list res(2); res[0] = ptr1; res[1] = ptr2; return res;});

}

//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) henon.cpp -o henon$(python3-config --extension-suffix)