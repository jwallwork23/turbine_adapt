from thetis import *
from turbine_adapt import *


__all__ = ["ErrorEstimator"]


class ErrorEstimator(object):
    """
    Error estimation for shallow water tidal
    turbine modelling applications.
    """

    @PETSc.Log.EventDecorator("ErrorEstimator.__init__")
    def __init__(
        self,
        options,
        mesh=None,
        norm_type="L2",
        error_estimator="difference_quotient",
        metric="isotropic_dwr",
        boundary=True,
    ):
        """
        :args options: :class:`FarmOptions` parameter object.
        :kwarg mesh: the mesh
        :kwarg norm_type: norm type for error estimator
        :kwarg error_estimator: error estimator type
        :kwarg metric: metric type
        :kwarg boundary: should boundary contributions be considered?
        """
        self.options = options
        self.mesh = mesh or options.mesh2d
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)
        self.P1_ten = TensorFunctionSpace(self.mesh, "CG", 1)
        self.h = CellSize(self.mesh)
        self.n = FacetNormal(self.mesh)
        self.dx = dx(domain=self.mesh)
        self.dS = dS(domain=self.mesh)
        self.ds = ds(domain=self.mesh)

        # Discretisation parameters
        assert self.options.swe_timestepper_type in ("CrankNicolson", "SteadyState")
        self.steady = self.options.swe_timestepper_type == "SteadyState"
        theta = self.options.swe_timestepper_options.implicitness_theta
        self.theta = None if self.steady else theta
        self.eta_is_dg = self.options.element_family == "dg-dg"

        # Get turbine drag coefficient
        self.drag_coefficient = self.options.quadratic_drag_coefficient
        Ct = self.options.corrected_thrust_coefficient * Constant(pi / 8)
        for i, subdomain_id in enumerate(options.farm_ids):
            indicator = interpolate(
                Constant(1.0), self.P0, subset=self.mesh.cell_subset(subdomain_id)
            )
            self.drag_coefficient = self.drag_coefficient + Ct * indicator

        # Error estimation parameters
        assert norm_type in ("L1", "L2")
        self.norm_type = norm_type
        assert error_estimator == "difference_quotient"
        self.error_estimator = error_estimator
        if metric not in ["isotropic_dwr", "anisotropic_dwr", "weighted_hessian"]:
            raise ValueError(f"Metric type {metric} not supported")
        self.metric_type = metric
        self.boundary = boundary

    def _Psi_uv_steady(self, uv, elev):
        """
        Strong residual for the velocity component
        in steady state mode.

        :arg uv: velocity field
        :arg elev: elevation field
        """
        H = self.options.bathymetry2d + elev
        g = physical_constants["g_grav"]
        nu = self.options.horizontal_viscosity
        Cd = self.drag_coefficient
        adv = dot(uv, nabla_grad(uv))
        vis = div(nu * grad(uv))
        return adv + g * grad(elev) + Cd * sqrt(inner(uv, uv)) * uv / H - vis

    def _Psi_u_steady(self, uv, elev):
        """
        Strong residual for the x-velocity component
        in steady state mode.

        :arg uv: velocity field
        :arg elev: elevation field
        """
        return self._Psi_uv_steady(uv, elev)[0]

    def _Psi_v_steady(self, uv, elev):
        """
        Strong residual for the y-velocity component
        in steady state mode.

        :arg uv: velocity field
        :arg elev: elevation field
        """
        return self._Psi_uv_steady(uv, elev)[1]

    def _Psi_eta_steady(self, uv, elev):
        """
        Strong residual for the free surface elevation
        component in steady state mode.

        :arg uv: velocity field
        :arg elev: elevation field
        """
        H = self.options.bathymetry2d + elev
        return div(H * uv)

    def _restrict(self, v):
        """
        Restrict an expression `v` over facets, according
        to the :attr:`norm_type`.

        :arg v: the UFL expression
        """
        if self.norm_type == "L1":
            return jump(abs(v), self.p0test)
        elif self.norm_type == "L2":
            return jump(v * v, self.p0test)
        else:
            raise NotImplementedError

    def _get_bnd_functions(self, eta_in, uv_in, bnd_in):
        """
        Get boundary contributions on segment `bnd_in`.

        :arg eta_in: the current elevation
        :arg uv_in: the current velocity
        :arg bnd_in: the boundary marker
        """
        funcs = self.options.bnd_conditions.get(bnd_in)
        if "elev" in funcs and "uv" in funcs:
            eta_ext = funcs["elev"]
            uv_ext = funcs["uv"]
        elif "elev" in funcs and "un" in funcs:
            eta_ext = funcs["elev"]
            uv_ext = funcs["un"] * self.n
        elif "elev" in funcs:
            eta_ext = funcs["elev"]
            uv_ext = uv_in
        elif "uv" in funcs:
            eta_ext = eta_in
            uv_ext = funcs["uv"]
        elif "un" in funcs:
            eta_ext = eta_in
            uv_ext = funcs["un"] * self.n
        else:
            raise Exception(f"Unsupported bnd type: {funcs.keys()}")
        return eta_ext, uv_ext

    def _psi_u_steady(self, *args):
        """
        Inter-element flux contributions from the x-velocity
        component for steady-state mode.
        """
        if self.steady:
            assert len(args) == 2
            uv, elev = args
            uv_old, elev_old = uv, elev
        else:
            assert len(args) == 4
            uv, elev, uv_old, elev_old = args
        psi = Function(self.P0)
        H = self.options.bathymetry2d + elev
        H_old = self.options.bathymetry2d + elev_old
        g = physical_constants["g_grav"]

        # Advection
        flux_terms = avg(uv_old[0]) * jump(uv, self.n)
        ibp_terms = self._restrict(uv[0] * dot(uv, self.n)) * self.dS
        uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
        if self.options.use_lax_friedrichs_velocity:
            gamma = 0.5 * abs(dot(avg(uv_old), self.n("-"))) * uv_lax_friedrichs
            flux_terms += gamma * jump(uv[0])

        # External pressure gradient
        if self.eta_is_dg:
            head_star = avg(elev_old) + sqrt(avg(H) / g) * jump(uv, self.n)
            flux_terms += -head_star * g
            ibp_terms += self._restrict(g * elev * self.n[0]) * self.dS

        # Viscosity
        p = self.options.polynomial_degree
        cp = (p + 1) * (p + 2) / 2
        l_normal = CellVolume(self.mesh) / FacetArea(self.mesh)
        sigma = self.options.sipg_factor * cp * l_normal
        sigma_max = max_value(sigma("+"), sigma("-"))
        nu = self.options.horizontal_viscosity
        if self.options.use_grad_div_viscosity_term:
            stress = 2 * nu * sym(grad(uv))
            stress_jump = 2 * avg(nu) * sym(tensor_jump(uv, self.n))
        else:
            stress = nu * grad(uv)
            stress_jump = avg(nu) * tensor_jump(uv, self.n)
        flux_terms += sigma_max * dot(stress_jump, jump(self.n))[0]
        # TODO: What about symmetrisation term?
        flux_terms += dot(avg(stress), jump(self.n))[0]

        # Boundary terms
        bnd_terms = {}
        if self.boundary:
            bnd_conditions = self.options.bnd_conditions
            for bnd_marker in bnd_conditions:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                eta_ext, uv_ext = self._get_bnd_functions(elev, uv, bnd_marker)
                eta_ext_old, uv_ext_old = self._get_bnd_functions(
                    elev_old, uv_old, bnd_marker
                )

                # External pressure gradient
                un_jump = inner(uv - uv_ext, self.n)
                if funcs is not None:
                    eta_rie = 0.5 * (elev + eta_ext) + sqrt(H / g) * un_jump
                elif self.eta_is_dg:
                    eta_rie = elev + sqrt(H / g) * un_jump
                bnd_terms[ds_bnd] = -g * eta_rie * self.n[0]
                if not self.eta_is_dg:
                    bnd_terms[ds_bnd] += g * eta * self.n[0]

                # Advection
                if funcs is not None:
                    eta_jump = elev_old - eta_ext_old
                    un_rie = (
                        0.5 * inner(uv_old + uv_ext_old, self.n)
                        + sqrt(g / H_old) * eta_jump
                    )
                    bnd_terms[ds_bnd] += -un_rie * 0.5 * (uv_ext[0] + uv[0])
                elif self.options.use_lax_friedrichs_velocity:
                    gamma = 0.5 * abs(dot(uv_old, self.n)) * uv_lax_friedrichs
                    bnd_terms[ds_bnd] += -gamma * 2 * dot(uv, self.n) * self.n[0]

                # Viscosity
                if funcs is not None:
                    if "un" in funcs:
                        delta_uv = (dot(uv, self.n) - funcs["un"]) * self.n[0]
                    else:
                        if uv_ext is uv:
                            continue
                        delta_uv = (uv - uv_ext)[0]
                    if self.options.use_grad_div_viscosity_term:
                        stress_jump = 2.0 * nu * sym(outer(delta_uv, self.n))
                    else:
                        stress_jump = nu * outer(outer(delta_uv, self.n))
                    bnd_terms[ds_bnd] += -sigma * nu * delta_uv
                    # TODO: What about symmetrisation term?

        # Compute flux norm
        mass_term = self.p0test * self.p0trial * self.dx
        if self.norm_type == "L1":
            flux_terms = 2 * avg(self.p0test) * abs(flux_terms) * self.dS
            bnd_terms = sum(
                self.p0test * abs(term) * ds_bnd for ds_bnd, term in bnd_terms.items()
            )
        elif self.norm_type == "L2":
            flux_terms = 2 * avg(self.p0test) * flux_terms * flux_terms * self.dS
            bnd_terms = sum(
                self.p0test * term * term * ds_bnd for ds_bnd, term in bnd_terms.items()
            )
        else:
            raise NotImplementedError
        sp = {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
        solve(
            mass_term == flux_terms + ibp_terms + bnd_terms, psi, solver_parameters=sp
        )
        psi.interpolate(abs(psi))
        return sqrt(psi) if self.norm_type == "L2" else psi

    def _psi_v_steady(self, *args):
        """
        Inter-element flux contributions from the y-velocity
        component for steady-state mode.
        """
        if self.steady:
            assert len(args) == 2
            uv, elev = args
            uv_old, elev_old = uv, elev
        else:
            assert len(args) == 4
            uv, elev, uv_old, elev_old = args
        psi = Function(self.P0)
        H = self.options.bathymetry2d + elev_old
        g = physical_constants["g_grav"]

        # Advection
        flux_terms = avg(uv[1]) * jump(uv, self.n)
        ibp_terms = self._restrict(uv[1] * dot(uv, self.n)) * self.dS
        uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
        if self.options.use_lax_friedrichs_velocity:
            gamma = 0.5 * abs(dot(avg(uv_old), self.n("-"))) * uv_lax_friedrichs
            flux_terms += gamma * jump(uv[1])

        # External pressure gradient
        if self.eta_is_dg:
            head_star = avg(elev_old) + sqrt(avg(H) / g) * jump(uv, self.n)
            flux_terms += -head_star * g
            ibp_terms += self._restrict(g * elev * self.n[1]) * self.dS

        # Viscosity
        p = self.options.polynomial_degree
        cp = (p + 1) * (p + 2) / 2
        l_normal = CellVolume(self.mesh) / FacetArea(self.mesh)
        sigma = self.options.sipg_factor * cp * l_normal
        sigma_max = max_value(sigma("+"), sigma("-"))
        nu = self.options.horizontal_viscosity
        if self.options.use_grad_div_viscosity_term:
            stress = 2 * nu * sym(grad(uv))
            stress_jump = 2 * avg(nu) * sym(tensor_jump(uv, self.n))
        else:
            stress = nu * grad(uv)
            stress_jump = avg(nu) * tensor_jump(uv, self.n)
        flux_terms += sigma_max * dot(stress_jump, jump(self.n))[1]
        # TODO: What about symmetrisation term?
        flux_terms += dot(avg(stress), jump(self.n))[1]

        # Boundary terms
        bnd_terms = {}
        if self.boundary:
            bnd_conditions = self.options.bnd_conditions
            for bnd_marker in bnd_conditions:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                eta_ext, uv_ext = self._get_bnd_functions(elev, uv, bnd_marker)
                eta_ext_old, uv_ext_old = self._get_bnd_functions(
                    elev_old, uv_old, bnd_marker
                )

                # External pressure gradient
                un_jump = inner(uv - uv_ext, self.n)
                if funcs is not None:
                    eta_rie = 0.5 * (elev + eta_ext) + sqrt(H / g) * un_jump
                elif self.eta_is_dg:
                    eta_rie = elev + sqrt(H / g) * un_jump
                bnd_terms[ds_bnd] = -g * eta_rie * self.n[1]
                if not self.eta_is_dg:
                    bnd_terms[ds_bnd] += g * eta * self.n[1]

                # Advection
                if funcs is not None:
                    eta_jump = elev_old - eta_ext_old
                    un_rie = (
                        0.5 * inner(uv_old + uv_ext_old, self.n)
                        + sqrt(g / H) * eta_jump
                    )
                    bnd_terms[ds_bnd] += -un_rie * 0.5 * (uv_ext[1] + uv[1])
                elif self.options.use_lax_friedrichs_velocity:
                    gamma = 0.5 * abs(dot(uv_old, self.n)) * uv_lax_friedrichs
                    bnd_terms[ds_bnd] += -gamma * 2 * dot(uv, self.n) * self.n[1]

                # Viscosity
                if funcs is not None:
                    if "un" in funcs:
                        delta_uv = (dot(uv, self.n) - funcs["un"]) * self.n[1]
                    else:
                        if uv_ext is uv:
                            continue
                        delta_uv = (uv - uv_ext)[1]
                    if self.options.use_grad_div_viscosity_term:
                        stress_jump = 2.0 * nu * sym(outer(delta_uv, self.n))
                    else:
                        stress_jump = nu * outer(outer(delta_uv, self.n))
                    bnd_terms[ds_bnd] += -sigma * nu * delta_uv
                    # TODO: What about symmetrisation term?

        # Compute flux norm
        mass_term = self.p0test * self.p0trial * self.dx
        if self.norm_type == "L1":
            flux_terms = 2 * avg(self.p0test) * abs(flux_terms) * self.dS
            bnd_terms = sum(
                self.p0test * abs(term) * ds_bnd for ds_bnd, term in bnd_terms.items()
            )
        elif self.norm_type == "L2":
            flux_terms = 2 * avg(self.p0test) * flux_terms * flux_terms * self.dS
            bnd_terms = sum(
                self.p0test * term * term * ds_bnd for ds_bnd, term in bnd_terms.items()
            )
        else:
            raise NotImplementedError
        sp = {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
        solve(
            mass_term == flux_terms + ibp_terms + bnd_terms, psi, solver_parameters=sp
        )
        psi.interpolate(abs(psi))
        return sqrt(psi) if self.norm_type == "L2" else psi

    def _psi_eta_steady(self, *args):
        """
        Inter-element flux contributions from the free surface
        elevation component for steady-state mode.
        """
        if self.steady:
            assert len(args) == 2
            uv, elev = args
            uv_old, elev_old = uv, elev
        else:
            assert len(args) == 4
            uv, elev, uv_old, elev_old = args
        psi = Function(self.P0)
        b = self.options.bathymetry2d
        H = b + elev_old
        g = physical_constants["g_grav"]
        flux_terms = 0

        # HUDiv
        ibp_terms = self._restrict(inner(H * uv, self.n)) * dS
        if self.eta_is_dg:
            un_rie = avg(dot(uv, self.n)) + sqrt(g / avg(H)) * jump(
                elev * self.n, self.n
            )
            flux_terms += -avg(H) * un_rie
        bnd_terms = {}
        if self.boundary:
            bnd_conditions = self.options.bnd_conditions
            for bnd_marker in bnd_conditions:
                funcs = bnd_conditions.get(bnd_marker)
                ds_bnd = self.ds(int(bnd_marker))
                eta_ext, uv_ext = self._get_bnd_functions(elev, uv, bnd_marker)
                eta_ext_old, uv_ext_old = self._get_bnd_functions(
                    elev_old, uv_old, bnd_marker
                )
                if funcs is not None:
                    h_av = 0.5 * (H + eta_ext_old + b)
                    eta_jump = elev - eta_ext
                    un_rie = (
                        0.5 * inner(uv + uv_ext, self.n) + sqrt(g / h_av) * eta_jump
                    )
                    un_jump = inner(uv_old - uv_ext_old, self.n)
                    eta_rie = 0.5 * (elev_old + eta_ext_old) + sqrt(h_av / g) * un_jump
                    h_rie = b + eta_rie
                    bnd_terms[ds_bnd] = -h_rie * un_rie

        # Compute flux norm
        mass_term = self.p0test * self.p0trial * self.dx
        if self.norm_type == "L1":
            flux_terms = 2 * avg(self.p0test) * abs(flux_terms) * self.dS
            bnd_terms = sum(
                self.p0test * abs(term) * ds_bnd for ds_bnd, term in bnd_terms.items()
            )
        elif self.norm_type == "L2":
            flux_terms = 2 * avg(self.p0test) * flux_terms * flux_terms * self.dS
            bnd_terms = sum(
                self.p0test * term * term * ds_bnd for ds_bnd, term in bnd_terms.items()
            )
        else:
            raise NotImplementedError
        sp = {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
        solve(
            mass_term == flux_terms + ibp_terms + bnd_terms, psi, solver_parameters=sp
        )
        psi.interpolate(abs(psi))
        return sqrt(psi) if self.norm_type == "L2" else psi

    def _Psi_uv_unsteady(self, uv, elev, uv_old, elev_old):
        """
        Strong residual for the velocity component
        in unsteady mode.

        :arg uv: velocity field at current timestep
        :arg elev: elevation field at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg elev_old: elevation field at previous timestep
        """
        f_time = (uv - uv_old) / self.options.timestep
        f = self._Psi_uv_steady(uv, elev)
        f_old = self._Psi_uv_steady(uv_old, elev_old)
        return f_time + self.theta * f + (1 - self.theta) * f_old

    def _Psi_u_unsteady(self, *args):
        """
        Strong residual for the x-velocity component
        in unsteady mode.

        :arg uv: velocity field at current timestep
        :arg elev: elevation field at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg elev_old: elevation field at previous timestep
        """
        return self._Psi_uv_unsteady(*args)[0]

    def _Psi_v_unsteady(self, *args):
        """
        Strong residual for the y-velocity component
        in unsteady mode.

        :arg uv: velocity field at current timestep
        :arg elev: elevation field at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg elev_old: elevation field at previous timestep
        """
        return self._Psi_uv_unsteady(*args)[1]

    def _Psi_eta_unsteady(self, uv, elev, uv_old, elev_old):
        """
        Strong residual for the free surface elevation
        component in unsteady mode.

        :arg uv: velocity field at current timestep
        :arg elev: elevation field at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg elev_old: elevation field at previous timestep
        """
        f_time = (elev - elev_old) / self.options.timestep
        f = self._Psi_eta_steady(uv, elev)
        f_old = self._Psi_eta_steady(uv_old, elev_old)
        return f_time + self.theta * f + (1 - self.theta) * f_old

    def _psi_u_unsteady(self, uv, elev, uv_old, elev_old):
        """
        Inter-element flux terms for the x-velocity component
        in unsteady mode.

        :arg uv: velocity field at current timestep
        :arg elev: elevation field at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg elev_old: elevation field at previous timestep
        """
        f = self._psi_u_steady(uv, elev, uv, elev)  # NOTE: Not semi-implicit
        f_old = self._psi_u_steady(uv_old, elev_old, uv_old, elev_old)
        return self.theta * f + (1 - self.theta) * f_old

    def _psi_v_unsteady(self, uv, elev, uv_old, elev_old):
        """
        Inter-element flux terms for the y-velocity component
        in unsteady mode.

        :arg uv: velocity field at current timestep
        :arg elev: elevation field at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg elev_old: elevation field at previous timestep
        """
        f = self._psi_v_steady(uv, elev, uv, elev)  # NOTE: Not semi-implicit
        f_old = self._psi_v_steady(uv_old, elev_old, uv_old, elev_old)
        return self.theta * f + (1 - self.theta) * f_old

    def _psi_eta_unsteady(self, uv, elev, uv_old, elev_old):
        """
        Inter-element flux terms for the free surface elevation
        component in unsteady mode.

        :arg uv: velocity field at current timestep
        :arg elev: elevation field at current timestep
        :arg uv_old: velocity field at previous timestep
        :arg elev_old: elevation field at previous timestep
        """
        f = self._psi_eta_steady(uv, elev, uv, elev)  # NOTE: Not semi-implicit
        f_old = self._psi_eta_steady(uv_old, elev_old, uv_old, elev_old)
        return self.theta * f + (1 - self.theta) * f_old

    @PETSc.Log.EventDecorator("ErrorEstimator.strong_residuals")
    def strong_residuals(self, *args):
        """
        Compute the strong residual over a single
        timestep, given current solution and
        lagged solution.
        """
        nargs = len(args)
        assert nargs == 2 if self.steady else 4
        if self.steady:
            Psi_uv = self._Psi_uv_steady(*args)
            Psi_eta = self._Psi_eta_steady(*args)
        else:
            Psi_uv = self._Psi_uv_unsteady(*args)
            Psi_eta = self._Psi_eta_unsteady(*args)
        if self.norm_type == "L1":
            return [
                assemble(self.p0test * abs(Psi_uv[0]) * self.dx),
                assemble(self.p0test * abs(Psi_uv[1]) * self.dx),
                assemble(self.p0test * abs(Psi_eta) * self.dx),
            ]
        elif self.norm_type == "L2":
            return [
                sqrt(abs(assemble(self.p0test * Psi_uv[0] * Psi_uv[0] * self.dx))),
                sqrt(abs(assemble(self.p0test * Psi_uv[1] * Psi_uv[1] * self.dx))),
                sqrt(abs(assemble(self.p0test * Psi_eta * Psi_eta * self.dx))),
            ]
        else:
            raise NotImplementedError

    @PETSc.Log.EventDecorator("ErrorEstimator.flux_terms")
    def flux_terms(self, *args):
        """
        Compute flux jump terms over a single
        timestep, given current solution tuple
        and lagged solution tuple.
        """
        if self.steady:
            return [
                self._psi_u_steady(*args),
                self._psi_v_steady(*args),
                self._psi_eta_steady(*args),
            ]
        else:
            return [
                self._psi_u_unsteady(*args),
                self._psi_v_unsteady(*args),
                self._psi_eta_unsteady(*args),
            ]

    @PETSc.Log.EventDecorator("ErrorEstimator.recover_laplacians")
    def recover_laplacians(self, uv, elev):
        """
        Recover the Laplacian of solution
        tuple `(uv, elev)`.
        """
        P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        g, phi = TrialFunction(P1_vec), TestFunction(P1_vec)
        a = inner(phi, g) * self.dx
        sp = {
            "ksp_type": "gmres",
            "ksp_gmres_restart": 20,
            "ksp_rtol": 1.0e-05,
            "pc_type": "sor",
        }
        projections = [Function(P1_vec) for i in range(3)]
        for f, proj in zip((uv[0], uv[1], elev), projections):
            L = f * dot(phi, self.n) * self.ds - div(phi) * f * self.dx
            solve(a == L, proj, solver_parameters=sp)
        if self.norm_type == "L1":
            return [interpolate(abs(div(proj)), self.P0) for proj in projections]
        else:
            return [
                sqrt(interpolate(inner(div(proj), div(proj)), self.P0))
                for proj in projections
            ]

    @PETSc.Log.EventDecorator("ErrorEstimator.recover_hessians")
    def recover_hessians(self, uv, elev):
        """
        Recover the Hessian of solution
        tuple `(uv, elev)`.
        """
        return [
            hessian_metric(recover_hessian(uv[0], mesh=self.mesh)),
            hessian_metric(recover_hessian(uv[1], mesh=self.mesh)),
            hessian_metric(recover_hessian(elev, mesh=self.mesh)),
        ]

    @PETSc.Log.EventDecorator("ErrorEstimator.difference_quotient")
    def difference_quotient(self, *args, flux_form=False):
        """
        Evaluate the dual weighted residual
        error estimator in difference quotient
        formulation.
        """
        nargs = len(args)
        assert nargs == 4 if self.steady else 8
        nargs2 = nargs // 2
        nargs22 = 2 + nargs2

        # Terms for standard a posteriori error estimate
        self.residuals = self.strong_residuals(*args[:nargs2])
        self.fluxes = self.flux_terms(*args[:nargs2])

        # Weighting term for the adjoint
        if flux_form:
            self.weights = self.flux_terms(*args[nargs2:])
        else:
            self.weights = self.recover_laplacians(*args[nargs2:nargs22])
            if not self.steady:  # Average recovered Laplacians
                for R, R_old in zip(self.weights, self.recover_laplacians(*args[-2:])):
                    R += R_old
                    R *= 0.5

        # Combine the two
        dq = Function(self.P0, name="Difference quotient")
        expr = 0
        for Psi, psi, R in zip(self.residuals, self.fluxes, self.weights):
            expr += (Psi + psi / sqrt(self.h)) * R
        dq.project(expr)
        dq.interpolate(abs(dq))  # Ensure positivity
        return dq

    @PETSc.Log.EventDecorator("ErrorEstimator.error_indicator")
    def error_indicator(self, *args, **kwargs):
        """
        Evaluate the error indicator of choice.
        """
        if self.error_estimator == "difference_quotient":
            flux_form = kwargs.get("flux_form", False)
            return self.difference_quotient(*args, flux_form=flux_form)
        else:
            raise NotImplementedError  # TODO

    @PETSc.Log.EventDecorator("ErrorEstimator.metric")
    def metric(self, *args, **kwargs):
        """
        Construct the metric of choice.
        """
        if self.metric_type == "isotropic_dwr":
            return isotropic_metric(self.error_indicator(*args, **kwargs))
        elif self.metric_type in ("anisotropic_dwr", "weighted_hessian"):
            flux_form = kwargs.get("flux_form", False)
            kwargs["approach"] = self.metric_type

            nargs = len(args)
            assert nargs == 4 if self.steady else 8
            nargs2 = nargs // 2
            nargs22 = 2 + nargs2

            # Terms for standard a posteriori error estimate
            Psi = self.strong_residuals(*args[:nargs2])

            # Adjoint weighting for anisotropic DWR
            if self.metric_type == "anisotropic_dwr":
                psi = self.flux_terms(*args[:nargs2])
                ee = [Psi_i + psi_i / sqrt(self.h) for Psi_i, psi_i in zip(Psi, psi)]
                if flux_form:
                    R = self.flux_terms(*args[nargs2:])
                else:
                    R = self.recover_laplacians(*args[nargs2:nargs22])
                    if self.steady:
                        ee = [ee_i * R_i for ee_i, R_i in zip(ee, R)]
                    else:
                        for i, R_next_i in enumerate(self.recover_laplacians(*args[nargs22:])):
                            ee[i] *= 0.5 * (R[i] + R_next_i)
            else:
                ee = Psi
                if flux_form:
                    raise ValueError("Flux form is incompatible with WH.")
            ee = firedrake.project(sum(ee), self.P0)

            # Hessian weighting term
            istart = nargs2 if self.metric_type == "weighted_hessian" else 0
            iend = istart + 2
            H = self.recover_hessians(*args[istart:iend])
            if not self.steady:  # Average recovered Hessians
                istart += 2
                iend += 2
                for i, H_old_i in enumerate(self.recover_hessians(*args[istart:iend])):
                    H[i] = 0.5 * (H[i] + H_old_i)
            H = firedrake.interpolate(sum(H), self.P1_ten)

            # Combine the two
            kwargs.pop("error_indicator", None)
            return anisotropic_metric([ee], [H], target_space=self.P1_ten, **kwargs)
        else:
            raise ValueError(f"Metric type {self.metric_type} not supported")
