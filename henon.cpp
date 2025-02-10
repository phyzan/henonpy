#include "pyode.hpp"


using Tx = double;
using Tf = vec::StackArray<double, 4>;

Tf hhode(const Tx& t, const Tf& q, const Tx* args){
    return {q[2], q[3], -(std::pow(args[4], 2.)*q[0] + args[0]*(std::pow(q[1], 2.) + 3.*args[1]*std::pow(q[0], 2.) + 2.*args[2]*q[0]*q[1])), -(std::pow(args[5], 2.)*q[1] + args[0]*(2.*q[0]*q[1] + args[2]*std::pow(q[0], 2.) + 3.*args[3]*std::pow(q[1], 2.)))};
}


bool getcond(const Tx& t1, const Tx& t2, const Tf& f1, const Tf& f2){
    return f1[1] < 0 && f2[1] >= 0;
}


#pragma GCC visibility push(hidden)
class HenonHeilesOde : public PyOde<Tx, Tf> {

    public:
        HenonHeilesOde(): PyOde<Tx, Tf>(hhode) {}

        const PyOdeResult<Tx> poincare_solve(const py::tuple& py_ics, const Tx& x, const Tx& dx, const Tx& err, py::str method, const int max_frames, py::tuple pyargs, const bool display) {

            vec::HeapArray<Tx> args;
            Tx x0;
            ICS<Tx, Tf> ics;
            size_t nd, nt;
        
            if (!pyargs.empty()){
                args = toCPP_Array<vec::HeapArray<Tx>>(pyargs);
            }
            x0 = py_ics[0].cast<Tx>();
            Tf f0 = toCPP_Array<Tf>(py_ics[1]);
            ics = {x0, f0};
        
            OdeResult<Tx, Tf> res = ODE<Tx, Tf>::solve(ics, x, dx, err, method.cast<std::string>().c_str(), max_frames, &args, getcond, nullptr, display);
            vec::HeapArray<Tx> f_flat = flatten(res.f);
            nd = res.f[0].size();
            nt = res.f.size();
        
            PyOdeResult<Tx> odres{res.x, f_flat, to_numpy(res.x, {nt}), to_numpy(f_flat, {nt, nd}), res.diverges, res.runtime};
        
            return odres;

        }


        HenonHeilesOde newcopy() const{
            return HenonHeilesOde();
        }

};
#pragma GCC visibility pop



PYBIND11_MODULE(henon, m){

    py::class_<HenonHeilesOde>(m, "henon", py::module_local())
        .def("solve", &HenonHeilesOde::pysolve,
            py::arg("ics"),
            py::arg("t"),
            py::arg("dt"),
            py::kw_only(),
            py::arg("err") = 0.,
            py::arg("method") = py::str("RK4"),
            py::arg("max_frames") = -1,
            py::arg("args") = py::tuple(),
            py::arg("getcond") = py::none(),
            py::arg("breakcond") = py::none(),
            py::arg("display") = false)
        .def("psolve", &HenonHeilesOde::poincare_solve,
            py::arg("ics"),
            py::arg("t"),
            py::arg("dt"),
            py::kw_only(),
            py::arg("err") = 0.,
            py::arg("method") = py::str("RK4"),
            py::arg("max_frames") = -1,
            py::arg("args") = py::tuple(),
            py::arg("display") = false)
        .def("copy", &HenonHeilesOde::newcopy);


    py::class_<PyOdeResult<Tx>>(m, "OdeResult", py::module_local())
        .def_readwrite("var", &PyOdeResult<Tx>::x)
        .def_readwrite("func", &PyOdeResult<Tx>::f)
        .def_readwrite("runtime", &PyOdeResult<Tx>::runtime)
        .def_readwrite("diverges", &PyOdeResult<Tx>::diverges);

    m.def("ode", []() {
        return HenonHeilesOde();
    });


}