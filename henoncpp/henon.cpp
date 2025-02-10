#include "pyode.hpp"


using Tx = double;
using Tf = vec::StackArray<double, 4>;

Tf hhode(const Tx& t, const Tf& q, const Tx* args){
    std::cout << args[0] << " " << args[1] << " " << args[3] << " " << args[4] << " " << args[5] << " ";
    return {q[2], q[3], -(std::pow(args[4], 2.)*q[0] + args[0]*(std::pow(q[1], 2.) + 3.*args[1]*std::pow(q[0], 2.) + 2.*args[2]*q[0]*q[1])), -(std::pow(args[5], 2.)*q[1] + args[0]*(2.*q[0]*q[1] + args[2]*std::pow(q[0], 2.) + 3.*args[3]*std::pow(q[1], 2.)))};
}


bool getcond(const Tx& t1, const Tx& t2, const Tf& f1, const Tf& f2){
    return f1[1] < 0 && f2[1] >= 0;
}


#pragma GCC visibility push(hidden)
class HenonHeilesOde : public PyOde<Tx, Tf> {

    public:
        HenonHeilesOde(): PyOde<Tx, Tf>(hhode) {}

        const PyOdeResult<Tx> poincare_solve(const Tx& x, const Tx& dx, const double& err, py::str method = py::str("method"), const int max_frames=-1, py::tuple pyargs = py::tuple(), const bool display=false){

            vec::HeapArray<Tx> args = toCPP_Array<vec::HeapArray<Tx>>(pyargs);

            OdeResult<Tx, Tf> res = solve(x, dx, err, method.cast<std::string>().c_str(), max_frames, &args, getcond, nullptr, display);


            size_t nd = res.f[0].size();
            size_t nt = res.f.size();
            vec::HeapArray<Tx> f_flat(nd*nt, true);
            for (size_t i=0; i<nt; i++){
                for (size_t j=0; j<nd; j++){
                    f_flat[i*nd+j] = res.f[i][j];
                }
            }

            PyOdeResult<Tx> odres{res.x, f_flat, to_numpy(res.x, {nt}), to_numpy(f_flat, {nt, nd}), res.diverges, res.runtime};

            return odres;

        }


        HenonHeilesOde newcopy() const{
            return HenonHeilesOde();
        }

        HenonHeilesOde newclone() const{
            
            HenonHeilesOde res = newcopy();
            py::list f0;
            for (size_t i=0; i< ics->f0.size(); i++){
                f0.append(ics->f0[i]);
            }
            res.set_ics(ics->x0, f0);
            return res;
        }

};
#pragma GCC visibility pop



PYBIND11_MODULE(HenonHeilsModule, m){

    py::class_<HenonHeilesOde>(m, "HenonHeilesOde", py::module_local())
        .def("solve", &HenonHeilesOde::pysolve,
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
            py::arg("t"),
            py::arg("dt"),
            py::kw_only(),
            py::arg("err") = 0.,
            py::arg("method") = py::str("RK4"),
            py::arg("max_frames") = -1,
            py::arg("args") = py::tuple(),
            py::arg("display") = false)
        .def("set_ics", &HenonHeilesOde::set_ics,
            py::arg("t0"),
            py::arg("f0"))
        .def("IntegrateAll", &HenonHeilesOde::IntAll,
            py::arg("ics"),
            py::arg("t"),
            py::arg("dt"),
            py::kw_only(),
            py::arg("err") = 0.,
            py::arg("method") = py::str("RK4"),
            py::arg("max_frames") = -1,
            py::arg("args") = py::tuple(),
            py::arg("threads") = -1)
        .def("copy", &HenonHeilesOde::newcopy)
        .def("clone", &HenonHeilesOde::newclone);


    py::class_<PyOdeResult<Tx>>(m, "OdeResult", py::module_local())
        .def_readwrite("var", &PyOdeResult<Tx>::x)
        .def_readwrite("func", &PyOdeResult<Tx>::f)
        .def_readwrite("runtime", &PyOdeResult<Tx>::runtime)
        .def_readwrite("diverges", &PyOdeResult<Tx>::diverges);

    m.def("ode", []() {
        return HenonHeilesOde();
    });


}