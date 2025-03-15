#include <odepack/pyode.hpp>


using Tt = double;
using Tf = vec<double, 4>;

Tf hhode(const Tt& t, const Tf& q, const std::vector<Tt>& args){
    return {q[2], q[3], -(std::pow(args[4], 2.)*q[0] + args[0]*(std::pow(q[1], 2.) + 3.*args[1]*std::pow(q[0], 2.) + 2.*args[2]*q[0]*q[1])), -(std::pow(args[5], 2.)*q[1] + args[0]*(2.*q[0]*q[1] + args[2]*std::pow(q[0], 2.) + 3.*args[3]*std::pow(q[1], 2.)))};
}


Tt event(const Tt& t, const Tf& q, const std::vector<Tt>& args){
    return q[1];
}

bool check_if(const Tt& t, const Tf& q, const std::vector<Tt>& args){
    return q[3] > 0;
}


#pragma GCC visibility push(hidden)
class HenonHeilesOde : public PyODE<Tt, Tf> {

    public:
        HenonHeilesOde(const py::array q0, const py::tuple args, const Tt stepsize, const Tt rtol, const Tt atol, const Tt min_step, const Tt event_tol):PyODE<Tt, Tf>(hhode, 0., toCPP_Array<Tt, Tf>(q0), stepsize, rtol, atol, min_step, toCPP_Array<Tt, std::vector<Tt>>(args), "RK45", event_tol, {Event<Tt, Tf>("Poincare Section", event, check_if)}){}

};
#pragma GCC visibility pop



PYBIND11_MODULE(henon, m){

    define_ode_module<Tt, Tf>(m);

    py::class_<HenonHeilesOde, PyODE<Tt, Tf>>(m, "HenonOde", py::module_local())
        .def(py::init<py::array, py::tuple, Tt, Tt, Tt, Tt, Tt>(),
                py::arg("q0"),
                py::arg("args"),
                py::arg("stepsize"),
                py::kw_only(),
                py::arg("rtol")=1e-6,
                py::arg("atol")=1e-12,
                py::arg("min_step")=0.,
                py::arg("event_tol")=1e-12);

}

//g++ -O3 -Wall -shared -std=c++20 -fopenmp -I/usr/include/python3.12 -I/usr/include/pybind11 -fPIC $(python3 -m pybind11 --includes) henon.cpp -o henon$(python3-config --extension-suffix)