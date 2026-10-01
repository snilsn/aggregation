// Python bindings for the snowagg C++ core.
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "snowagg/aggregate.hpp"
#include "snowagg/crystal.hpp"
#include "snowagg/dendrite.hpp"
#include "snowagg/deposition.hpp"
#include "snowagg/generator.hpp"
#include "snowagg/mcs.hpp"
#include "snowagg/rng.hpp"
#include "snowagg/rotator.hpp"
#include "snowagg/stl.hpp"

namespace py = pybind11;
using namespace snowagg;

using DArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;
using I32Array = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

namespace {

Points to_points(const DArray& a) {
    if (a.ndim() != 2 || a.shape(1) != 3) throw std::invalid_argument("expected an (N,3) array");
    const auto n = a.shape(0);
    Points X(n);
    auto r = a.unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) X[i] = {r(i, 0), r(i, 1), r(i, 2)};
    return X;
}

py::array_t<double> from_points(const Points& X) {
    py::array_t<double> a({static_cast<py::ssize_t>(X.size()), static_cast<py::ssize_t>(3)});
    auto w = a.mutable_unchecked<2>();
    for (size_t i = 0; i < X.size(); ++i)
        for (int d = 0; d < 3; ++d) w(i, d) = X[i][d];
    return a;
}

py::array_t<int64_t> from_ipoints(const std::vector<IVec3>& X) {
    py::array_t<int64_t> a({static_cast<py::ssize_t>(X.size()), static_cast<py::ssize_t>(3)});
    auto w = a.mutable_unchecked<2>();
    for (size_t i = 0; i < X.size(); ++i)
        for (int d = 0; d < 3; ++d) w(i, d) = X[i][d];
    return a;
}

std::vector<IVec3> to_ipoints(const IArray& a) {
    if (a.ndim() != 2 || a.shape(1) != 3) throw std::invalid_argument("expected an (N,3) array");
    std::vector<IVec3> X(a.shape(0));
    auto r = a.unchecked<2>();
    for (py::ssize_t i = 0; i < a.shape(0); ++i) X[i] = {r(i, 0), r(i, 1), r(i, 2)};
    return X;
}

std::vector<int32_t> to_ident(const py::object& ident, size_t n) {
    if (py::isinstance<py::int_>(ident)) return std::vector<int32_t>(n, ident.cast<int32_t>());
    auto a = ident.cast<I32Array>();
    if (a.ndim() == 0) return std::vector<int32_t>(n, *a.data());
    if (static_cast<size_t>(a.size()) != n) throw std::invalid_argument("ident has the wrong length");
    return std::vector<int32_t>(a.data(), a.data() + n);
}

py::array_t<double> mat_to_array(const Mat3& M) {
    py::array_t<double> a({3, 3});
    auto w = a.mutable_unchecked<2>();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) w(i, j) = M[i][j];
    return a;
}

py::array_t<uint8_t> proj_to_array(const ProjGrid& pg) {
    py::array_t<uint8_t> a({static_cast<py::ssize_t>(pg.nx), static_cast<py::ssize_t>(pg.ny)});
    std::copy(pg.data.begin(), pg.data.end(), a.mutable_data());
    return a;
}

py::list extent_to_list(const Extent& e) {
    py::list l;
    for (int d = 0; d < 3; ++d) {
        py::list pair;
        pair.append(e(d, 0));
        pair.append(e(d, 1));
        l.append(pair);
    }
    return l;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of snowagg (port of the aggregation snowflake model)";

    m.def("set_num_threads", [](int n) {
#ifdef _OPENMP
        omp_set_num_threads(std::max(1, n));
#else
        (void)n;
#endif
    });
    m.def("get_num_threads", []() {
#ifdef _OPENMP
        return omp_get_max_threads();
#else
        return 1;
#endif
    });

    // ---- RNG --------------------------------------------------------------
    py::class_<Rng>(m, "Rng", "MT19937 generator bit-compatible with numpy's legacy RandomState")
        .def(py::init<uint32_t>(), py::arg("seed") = 0)
        .def("seed", &Rng::seed)
        .def("seed_random", &Rng::seed_random)
        .def("rand", &Rng::rand)
        .def("randint", &Rng::randint)
        .def("standard_exponential", &Rng::standard_exponential)
        .def("gauss", &Rng::gauss)
        .def("interval", &Rng::interval)
        .def("next_u32", &Rng::next_u32)
        .def("get_state",
             [](const Rng& r) {
                 py::array_t<uint32_t> key(Rng::N);
                 std::copy(r.key.begin(), r.key.end(), key.mutable_data());
                 return py::make_tuple("MT19937", key, r.pos, static_cast<int>(r.has_gauss), r.gauss_cache);
             })
        .def("set_state", [](Rng& r, const py::tuple& st) {
            if (st.size() < 3 || st[0].cast<std::string>() != "MT19937")
                throw std::invalid_argument("expected a legacy numpy MT19937 state tuple");
            auto key = st[1].cast<py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>();
            if (key.size() != Rng::N) throw std::invalid_argument("state key must have 624 entries");
            std::copy(key.data(), key.data() + Rng::N, r.key.begin());
            r.pos = st[2].cast<int>();
            r.has_gauss = st.size() > 3 ? st[3].cast<int>() != 0 : false;
            r.gauss_cache = st.size() > 4 ? st[4].cast<double>() : 0.0;
        });

    // ---- crystals ---------------------------------------------------------
    py::class_<Crystal, std::shared_ptr<Crystal>>(m, "Crystal")
        .def("max_radius", &Crystal::max_radius)
        .def("name", &Crystal::name)
        .def("is_inside", [](const Crystal& c, const DArray& x, const DArray& y, const DArray& z) {
            if (x.size() != y.size() || x.size() != z.size())
                throw std::invalid_argument("x, y, z must have the same size");
            std::vector<py::ssize_t> shape(x.shape(), x.shape() + x.ndim());
            py::array_t<bool> out(shape);
            bool* o = out.mutable_data();
            const double *px = x.data(), *py_ = y.data(), *pz = z.data();
            for (py::ssize_t i = 0; i < x.size(); ++i) o[i] = c.is_inside(px[i], py_[i], pz[i]);
            return out;
        });
    py::class_<HexPrism, Crystal, std::shared_ptr<HexPrism>>(m, "HexPrism")
        .def_readonly("D", &HexPrism::D)
        .def_readonly("a", &HexPrism::a)
        .def_readonly("L", &HexPrism::L)
        .def_readonly("V", &HexPrism::V)
        .def_readonly("A", &HexPrism::A)
        .def_readonly("r", &HexPrism::r)
        .def_property_readonly("centers", [](const HexPrism& h) {
            py::array_t<double> a({6, 2});
            auto w = a.mutable_unchecked<2>();
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 2; ++j) w(i, j) = h.centers[i][j];
            return a;
        });
    py::class_<Plate, HexPrism, std::shared_ptr<Plate>>(m, "Plate").def(py::init<double>(), py::arg("D"));
    py::class_<Column, HexPrism, std::shared_ptr<Column>>(m, "Column").def(py::init<double>(), py::arg("D"));
    py::class_<Needle, HexPrism, std::shared_ptr<Needle>>(m, "Needle").def(py::init<double>(), py::arg("D"));
    py::class_<Dendrite, HexPrism, std::shared_ptr<Dendrite>>(m, "Dendrite")
        .def(py::init([](double D, const DArray& hex_grid, int64_t grid_size) {
                 if (hex_grid.ndim() != 2) throw std::invalid_argument("hex_grid must be 2D");
                 std::vector<double> g(hex_grid.data(), hex_grid.data() + hex_grid.size());
                 return std::make_shared<Dendrite>(D, g, hex_grid.shape(0), hex_grid.shape(1), grid_size);
             }),
             py::arg("D"), py::arg("hex_grid"), py::arg("grid_size") = 400)
        .def_readonly("grid_size", &Dendrite::grid_size)
        .def_readonly("grid_D", &Dendrite::grid_D);
    py::class_<BulletBase, Crystal, std::shared_ptr<BulletBase>>(m, "BulletBase")
        .def_readonly("D", &BulletBase::D)
        .def_readonly("L", &BulletBase::L)
        .def_readonly("a", &BulletBase::a)
        .def_readonly("t", &BulletBase::t);
    py::class_<Rosette, BulletBase, std::shared_ptr<Rosette>>(m, "Rosette")
        .def(py::init<double, double>(), py::arg("D"), py::arg("tan_alpha") = NAN);
    py::class_<Bullet, BulletBase, std::shared_ptr<Bullet>>(m, "Bullet")
        .def(py::init<double, double>(), py::arg("D"), py::arg("tan_alpha") = NAN);
    py::class_<Spheroid, Crystal, std::shared_ptr<Spheroid>>(m, "Spheroid")
        .def(py::init<double, double>(), py::arg("D_max"), py::arg("axis_ratio") = 1.0)
        .def_readonly("a", &Spheroid::a)
        .def_readonly("c", &Spheroid::c)
        .def_readonly("axis_ratio", &Spheroid::axis_ratio);
    m.def("brentq", [](const std::function<double(double)>& f, double a, double b) { return brentq(f, a, b); });

    // ---- rotators ---------------------------------------------------------
    m.def("rotation_matrix", [](double a, double b, double p) { return mat_to_array(rotation_matrix(a, b, p)); });
    py::class_<Rotator, std::shared_ptr<Rotator>>(m, "Rotator")
        .def("rotate", [](const Rotator& r, const DArray& X, Rng& rng) {
            Points P = to_points(X);
            r.rotate(P, rng);
            return from_points(P);
        }, "Rotate an (N,3) array of points");
    py::class_<NullRotator, Rotator, std::shared_ptr<NullRotator>>(m, "NullRotator").def(py::init<>());
    py::class_<UniformRotator, Rotator, std::shared_ptr<UniformRotator>>(m, "UniformRotator").def(py::init<>());
    py::class_<HorizontalRotator, Rotator, std::shared_ptr<HorizontalRotator>>(m, "HorizontalRotator")
        .def(py::init<>());
    py::class_<SamplePDF>(m, "SamplePDF")
        .def(py::init<std::vector<double>, std::vector<double>>(), py::arg("x"), py::arg("Y"))
        .def("__call__", &SamplePDF::operator())
        .def_readonly("x", &SamplePDF::x)
        .def_readonly("Y", &SamplePDF::Y);
    py::class_<PartialAligningRotator, Rotator, std::shared_ptr<PartialAligningRotator>>(m, "PartialAligningRotator")
        .def(py::init<double, bool>(), py::arg("exp_sig_deg") = 40.0, py::arg("random_flip") = false)
        .def(py::init<double, bool, SamplePDF>(), py::arg("exp_sig_deg"), py::arg("random_flip"), py::arg("table"))
        .def_readonly("exp_sig", &PartialAligningRotator::exp_sig)
        .def_readonly("random_flip", &PartialAligningRotator::random_flip)
        .def_readonly("beta_sample", &PartialAligningRotator::beta_sample);

    // ---- generator --------------------------------------------------------
    py::class_<MonodisperseGenerator, std::shared_ptr<MonodisperseGenerator>>(m, "MonodisperseGenerator")
        .def(py::init([](std::shared_ptr<Crystal> c, std::shared_ptr<Rotator> r, double g) {
                 py::gil_scoped_release nogil;
                 return std::make_shared<MonodisperseGenerator>(c, r, g);
             }),
             py::arg("crystal"), py::arg("rot"), py::arg("grid_res"))
        .def_readonly("grid_res", &MonodisperseGenerator::grid_res)
        .def_readonly("max_r", &MonodisperseGenerator::max_r)
        .def_readonly("n", &MonodisperseGenerator::n)
        .def_property_readonly("lattice", [](const MonodisperseGenerator& g) { return from_points(g.lattice); })
        .def_property_readonly("num_elements", [](const MonodisperseGenerator& g) { return g.lattice.size(); })
        .def("generate", [](const MonodisperseGenerator& g, Rng& rng) { return from_points(g.generate(rng)); });

    // ---- aggregate --------------------------------------------------------
    py::class_<Aggregate, std::shared_ptr<Aggregate>>(m, "Aggregate")
        .def(py::init([](const DArray& X, double grid_res, int32_t ident, bool fortran_order) {
                 return std::make_shared<Aggregate>(to_points(X), grid_res, ident, fortran_order);
             }),
             py::arg("X"), py::arg("grid_res"), py::arg("ident") = 0, py::arg("fortran_order") = true)
        .def_readwrite("fortran_order", &Aggregate::fortran_order)
        .def(py::init([](const MonodisperseGenerator& gen, Rng& rng, int32_t ident) {
                 return std::make_shared<Aggregate>(gen, rng, ident);
             }),
             py::arg("generator"), py::arg("rng"), py::arg("ident") = 0)
        .def_readonly("grid_res", &Aggregate::grid_res)
        .def("__len__", &Aggregate::size)
        .def_property(
            "X", [](const Aggregate& a) { return from_points(a.X); },
            [](Aggregate& a, const DArray& X) {
                Points P = to_points(X);
                if (P.size() != a.ident.size()) a.ident.resize(P.size(), 0);
                a.X = std::move(P);
                a.fortran_order = false;
            })
        .def_property(
            "ident",
            [](const Aggregate& a) {
                py::array_t<int32_t> r(static_cast<py::ssize_t>(a.ident.size()));
                std::copy(a.ident.begin(), a.ident.end(), r.mutable_data());
                return r;
            },
            [](Aggregate& a, const py::object& v) { a.ident = to_ident(v, a.X.size()); })
        .def_property(
            "extent", [](const Aggregate& a) { return extent_to_list(a.extent); },
            [](Aggregate& a, const std::vector<std::vector<double>>& e) {
                if (e.size() != 3) throw std::invalid_argument("extent must be 3x2");
                for (int d = 0; d < 3; ++d) {
                    a.extent(d, 0) = e[d].at(0);
                    a.extent(d, 1) = e[d].at(1);
                }
            })
        .def("update_extent", &Aggregate::update_extent)
        .def("update_coordinates", &Aggregate::update_coordinates)
        .def("add_elements",
             [](Aggregate& a, const DArray& X, const py::object& ident, bool update, bool added_fortran) {
                 Points P = to_points(X);
                 a.add_elements(P, to_ident(ident, P.size()), update, added_fortran);
             },
             py::arg("added_elements"), py::arg("ident") = 0, py::arg("update") = true,
             py::arg("added_fortran") = false)
        .def("remove_elements",
             [](Aggregate& a, const DArray& X, double tol, bool update) {
                 a.remove_elements(to_points(X), tol, update);
             },
             py::arg("removed_elements"), py::arg("tolerance") = 0.001, py::arg("update") = true)
        .def("principal_axes", [](const Aggregate& a) { return mat_to_array(a.principal_axes()); })
        .def("aspect_ratio", &Aggregate::aspect_ratio)
        .def("align", &Aggregate::align, py::call_guard<py::gil_scoped_release>())
        .def("rotate", &Aggregate::rotate, py::arg("rotator"), py::arg("rng"),
             py::call_guard<py::gil_scoped_release>())
        .def("project_on_dim", [](const Aggregate& a, int dim) { return proj_to_array(a.project_on_dim(dim)); },
             py::arg("dim") = 2)
        .def("project_on_direction",
             [](Aggregate& a, double alpha, double beta) { return proj_to_array(a.project_on_direction(alpha, beta)); })
        .def("projected_area", &Aggregate::projected_area, py::arg("dim") = 2)
        .def_static("projected_aspect_ratio_of", [](const py::array_t<uint8_t, py::array::c_style>& p) {
            ProjGrid pg;
            pg.nx = p.shape(0);
            pg.ny = p.shape(1);
            pg.data.assign(p.data(), p.data() + p.size());
            return Aggregate::projected_aspect_ratio(pg);
        })
        .def("add_particle",
             [](Aggregate& a, const DArray& particle, const py::object& ident, bool required, double pen_depth,
                Rng& rng) {
                 Points P = to_points(particle);
                 auto id = to_ident(ident, P.size());
                 py::gil_scoped_release nogil;
                 return a.add_particle(P, id, required, pen_depth, rng);
             },
             py::arg("particle"), py::arg("ident"), py::arg("required"), py::arg("pen_depth"), py::arg("rng"))
        .def("add_aggregate",
             [](Aggregate& a, const Aggregate& other, bool required, double pen_depth, Rng& rng) {
                 py::gil_scoped_release nogil;
                 return a.add_particle(other.X, other.ident, required, pen_depth, rng);
             },
             "add_particle using another Aggregate's elements and identifiers", py::arg("other"),
             py::arg("required"), py::arg("pen_depth"), py::arg("rng"))
        .def("grid", [](const Aggregate& a, double res, Rng& rng) { return from_ipoints(a.grid(res, rng)); },
             py::arg("res"), py::arg("rng"))
        .def("add_rime_particles", &Aggregate::add_rime_particles, py::arg("N"), py::arg("pen_depth"),
             py::arg("compact_dist"), py::arg("rng"), py::call_guard<py::gil_scoped_release>())
        .def("compact_rime",
             [](const Aggregate& a, const std::array<double, 3>& X, const DArray& X_near, double max_dist,
                double min_move, double dr, int max_iters) {
                 return a.compact_rime(X, to_points(X_near), max_dist, min_move, dr, max_iters);
             },
             py::arg("X"), py::arg("X_near"), py::arg("max_dist") = 0., py::arg("min_move") = 0.01,
             py::arg("dr") = 0.1, py::arg("max_iters") = 100)
        .def("copy", [](const Aggregate& a) { return std::make_shared<Aggregate>(a); });

    // ---- free functions ---------------------------------------------------
    m.def("minimum_covering_sphere", [](const DArray& X) {
        Points P = to_points(X);
        std::pair<Vec3, double> r;
        {
            py::gil_scoped_release nogil;
            r = minimum_covering_sphere(P);
        }
        py::array_t<double> c(3);
        std::copy(r.first.begin(), r.first.end(), c.mutable_data());
        return py::make_tuple(c, r.second);
    });
    m.def("minimum_covering_sphere_agg", [](const Aggregate& a) {
        std::pair<Vec3, double> r;
        {
            py::gil_scoped_release nogil;
            r = minimum_covering_sphere(a.X);
        }
        py::array_t<double> c(3);
        std::copy(r.first.begin(), r.first.end(), c.mutable_data());
        return py::make_tuple(c, r.second);
    });
    m.def("generate_dendrite",
          [](double alpha, double beta, double gamma, int64_t grid_size, int64_t num_iter) {
              DendriteResult r;
              {
                  py::gil_scoped_release nogil;
                  r = generate_dendrite(alpha, beta, gamma, grid_size, num_iter);
              }
              py::array_t<double> g({r.size, r.size});
              std::copy(r.grid.begin(), r.grid.end(), g.mutable_data());
              return py::make_tuple(g, r.iterations, r.margin);
          },
          py::arg("alpha"), py::arg("beta"), py::arg("gamma"), py::arg("grid_size") = 1000,
          py::arg("num_iter") = 10000);
    m.def("grow_ice",
          [](Aggregate& agg, Rng& rng, double ice_vol, double outer_rad_norm, double move_norm,
             const py::object& acos_fn) {
              if (acos_fn.is_none()) {
                  py::gil_scoped_release nogil;
                  grow_ice(agg, rng, ice_vol, outer_rad_norm, move_norm);
              } else {
                  auto f = [&acos_fn](double v) { return acos_fn(v).cast<double>(); };
                  grow_ice(agg, rng, ice_vol, outer_rad_norm, move_norm, f);
              }
          },
          py::arg("agg"), py::arg("rng"), py::arg("ice_vol"), py::arg("outer_rad_norm") = 2.,
          py::arg("move_norm") = 1., py::arg("acos") = py::none());
    m.def("snowflake_grid_to_stl",
          [](const IArray& X, const std::string& fn) { snowflake_grid_to_stl(to_ipoints(X), fn); });
}
