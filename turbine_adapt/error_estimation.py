from turbine_adapt import *


__all__ = ["ErrorEstimator"]


class ErrorEstimator(object):
    """
    Error estimation for shallow water tidal
    turbine modelling applications.
    """
    def __init__(self, options, mesh=None, norm_type='L2'):
        """
        :args options: :class:`FarmOptions` parameter object.
        """
        self.options = options
        self.mesh = mesh or options.mesh2d
        self.P0 = FunctionSpace(self.mesh, "DG", 0)
        self.p0test = TestFunction(self.P0)
        self.p0trial = TrialFunction(self.P0)
        self.h = CellSize(self.mesh)
        self.n = FacetNormal(self.mesh)

        # Discretisation parameters
        assert self.options.timestepper_type in ('CrankNicolson', 'SteadyState')
        self.steady = self.options.timestepper_type == 'SteadyState'
        self.theta = None if self.steady else self.options.timestepper_options.implicitness_theta
        self.eta_is_dg = self.options.element_family == 'dg-dg'

        # Get turbine drag coefficient
        self.drag_coefficient = self.options.quadratic_drag_coefficient
        Ct = self.options.corrected_thrust_coefficient*Constant(pi/8)
        for i, subdomain_id in enumerate(options.farm_ids):
            indicator = interpolate(Constant(1.0), self.P0, subset=self.mesh.cell_subset(subdomain_id))
            self.drag_coefficient = self.drag_coefficient + Ct*indicator

        # Error estimation parameters
        assert norm_type in ('L1', 'L2')
        self.norm_type = norm_type

    def _Psi_u_steady(self, uv, elev):
        H = self.options.bathymetry2d + elev
        g = physical_constants['g_grav']
        nu = self.options.horizontal_viscosity
        Cd = self.drag_coefficient
        adv = dot(uv, nabla_grad(uv))
        vis = div(nu*grad(uv))
        return adv[0] + g*elev.dx(0) + Cd*sqrt(inner(uv, uv))*uv[0]/H - vis[0]

    def _Psi_v_steady(self, uv, elev):
        H = self.options.bathymetry2d + elev
        g = physical_constants['g_grav']
        nu = self.options.horizontal_viscosity
        Cd = self.drag_coefficient
        adv = dot(uv, nabla_grad(uv))
        vis = div(nu*grad(uv))
        return adv[1] + g*elev.dx(1) + Cd*sqrt(inner(uv, uv))*uv[1]/H - vis[1]

    def _Psi_eta_steady(self, uv, elev):
        H = self.options.bathymetry2d + elev
        return div(H*uv)

    def _restrict(self, v):
        if self.norm_type == 'L1':
            return jump(abs(v), self.p0test)
        else:
            return jump(v*v, self.p0test)

    def _psi_u_steady(self, uv, elev):
        psi = Function(self.P0)
        H = self.options.bathymetry2d + elev
        g = physical_constants['g_grav']
        flux_terms = 0
        ibp_terms = 0
        if self.eta_is_dg:

            # External pressure gradient
            head_star = avg(elev) + sqrt(avg(H)/g)*jump(uv, self.n)
            flux_terms += -head_star*g
            ibp_terms += g*elev*self.n[0]

        # Advection
        flux_terms += avg(uv[0])*jump(uv, self.n)
        ibp_terms += uv[0]*dot(uv, self.n)
        if self.options.use_lax_friedrichs_velocity:
            uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
            gamma = 0.5*abs(dot(avg(uv), self.n('-')))*uv_lax_friedrichs
            flux_terms += gamma*jump(uv[0])

        # Viscosity
        p = self.options.polynomial_degree
        cp = (p + 1)*(p + 2)/2
        l_normal = CellVolume(self.mesh)/FacetArea(self.mesh)
        sigma = self.options.sipg_factor*cp*l_normal
        sigma_max = max_value(sigma('+'), sigma('-'))
        nu = self.options.horizontal_viscosity
        if self.options.use_grad_div_viscosity_term:
            stress = 2*nu*sym(grad(uv))
            stress_jump = 2*avg(nu)*sym(tensor_jump(uv, self.n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, self.n)
        flux_terms += sigma_max*dot(stress_jump, jump(self.n))[0]
        # TODO: What about symmetrisation term?
        flux_terms += dot(avg(stress), jump(self.n))[0]

        # Compute flux norm
        mass_term = self.p0test*self.p0trial*dx
        ibp_terms = self._restrict(ibp_terms)*dS
        if self.norm_type == 'L1':
            flux_terms = 2*avg(self.p0test)*abs(flux_terms)*dS
        else:
            flux_terms = 2*avg(self.p0test)*flux_terms*flux_terms*dS
        sp = {'ksp_type': 'preonly', 'pc_type': 'jacobi'}
        solve(mass_term == flux_terms + ibp_terms, psi, solver_parameters=sp)
        return sqrt(psi) if self.norm_type == 'L2' else psi

    def _psi_v_steady(self, uv, elev):
        psi = Function(self.P0)
        H = self.options.bathymetry2d + elev
        g = physical_constants['g_grav']
        flux_terms = 0
        ibp_terms = 0
        if self.eta_is_dg:

            # External pressure gradient
            head_star = avg(elev) + sqrt(avg(H)/g)*jump(uv, self.n)
            flux_terms += -head_star*g
            ibp_terms += g*elev*self.n[1]

        # Advection
        flux_terms += avg(uv[1])*jump(uv, self.n)
        ibp_terms += uv[1]*dot(uv, self.n)
        if self.options.use_lax_friedrichs_velocity:
            uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
            gamma = 0.5*abs(dot(avg(uv), self.n('-')))*uv_lax_friedrichs
            flux_terms += gamma*jump(uv[1])

        # Viscosity
        p = self.options.polynomial_degree
        cp = (p + 1)*(p + 2)/2
        l_normal = CellVolume(self.mesh)/FacetArea(self.mesh)
        sigma = self.options.sipg_factor*cp*l_normal
        sigma_max = max_value(sigma('+'), sigma('-'))
        nu = self.options.horizontal_viscosity
        if self.options.use_grad_div_viscosity_term:
            stress = 2*nu*sym(grad(uv))
            stress_jump = 2*avg(nu)*sym(tensor_jump(uv, self.n))
        else:
            stress = nu*grad(uv)
            stress_jump = avg(nu)*tensor_jump(uv, self.n)
        flux_terms += sigma_max*dot(stress_jump, jump(self.n))[1]
        # TODO: What about symmetrisation term?
        flux_terms += dot(avg(stress), jump(self.n))[1]

        # Compute flux norm
        mass_term = self.p0test*self.p0trial*dx
        ibp_terms = self._restrict(ibp_terms)*dS
        if self.norm_type == 'L1':
            flux_terms = 2*avg(self.p0test)*abs(flux_terms)*dS
        else:
            flux_terms = 2*avg(self.p0test)*flux_terms*flux_terms*dS
        sp = {'ksp_type': 'preonly', 'pc_type': 'jacobi'}
        solve(mass_term == flux_terms + ibp_terms, psi, solver_parameters=sp)
        return sqrt(psi) if self.norm_type == 'L2' else psi

    def _psi_eta_steady(self, uv, elev):
        psi = Function(self.P0)
        H = self.options.bathymetry2d + elev
        g = physical_constants['g_grav']
        flux_terms = 0

        # HUDiv
        ibp_terms = inner(H*uv, self.n)
        if self.eta_is_dg:
            un_rie = avg(dot(uv, self.n)) + sqrt(g/avg(H))*jump(elev*self.n, self.n)
            flux_terms += -avg(H)*un_rie

        # Compute flux norm
        mass_term = self.p0test*self.p0trial*dx
        ibp_terms = self._restrict(ibp_terms)*dS
        if self.norm_type == 'L1':
            flux_terms = 2*avg(self.p0test)*abs(flux_terms)*dS
        else:
            flux_terms = 2*avg(self.p0test)*flux_terms*flux_terms*dS
        sp = {'ksp_type': 'preonly', 'pc_type': 'jacobi'}
        solve(mass_term == flux_terms + ibp_terms, psi, solver_parameters=sp)
        return sqrt(psi) if self.norm_type == 'L2' else psi

    def _Psi_u_unsteady(self, uv, elev, uv_old, elev_old):
        f_time = (uv[0] - uv_old[0])/self.options.timestep
        f = self._Psi_u_steady(uv, elev)
        f_old = self._Psi_u_steady(uv_old, elev_old)
        return f_time + self.theta*f + (1-self.theta)*f_old

    def _Psi_v_unsteady(self, uv, elev, uv_old, elev_old):
        f_time = (uv[1] - uv_old[1])/self.options.timestep
        f = self._Psi_v_steady(uv, elev)
        f_old = self._Psi_v_steady(uv_old, elev_old)
        return f_time + self.theta*f + (1-self.theta)*f_old

    def _Psi_eta_unsteady(self, uv, elev, uv_old, elev_old):
        f_time = (elev - elev_old)/self.options.timestep
        f = self._Psi_eta_steady(uv, elev)
        f_old = self._Psi_eta_steady(uv_old, elev_old)
        return f_time + self.theta*f + (1-self.theta)*f_old

    def _psi_u_unsteady(self, uv, elev, uv_old, elev_old):
        f = self._psi_u_steady(uv, elev)
        f_old = self._psi_u_steady(uv_old, elev_old)
        return self.theta*f + (1-self.theta)*f_old

    def _psi_v_unsteady(self, uv, elev, uv_old, elev_old):
        f = self._psi_v_steady(uv, elev)
        f_old = self._psi_v_steady(uv_old, elev_old)
        return self.theta*f + (1-self.theta)*f_old

    def _psi_eta_unsteady(self, uv, elev, uv_old, elev_old):
        f = self._psi_eta_steady(uv, elev)
        f_old = self._psi_eta_steady(uv_old, elev_old)
        return self.theta*f + (1-self.theta)*f_old

    def strong_residuals(self, uv, elev, uv_old, elev_old):
        """
        Compute the strong residual over a single
        timestep, given current solution tuple
        `(uv, elev)` and lagged solution tuple
        `(uv_old, elev_old)`.

        If :attr:`timestepper` is set to
        `'SteadyState'` then only the `uv` and
        `elev` arguments are used.
        """
        if self.steady:
            args = (uv, elev)
            Psi_u = self._Psi_u_steady(*args)
            Psi_v = self._Psi_v_steady(*args)
            Psi_eta = self._Psi_eta_steady(*args)
        else:
            args = (uv, elev, uv_old, elev_old)
            Psi_u = self._Psi_u_unsteady(*args)
            Psi_v = self._Psi_v_unsteady(*args)
            Psi_eta = self._Psi_eta_unsteady(*args)
        if self.norm_type == 'L1':
            return [
                assemble(self.p0test*abs(Psi_u)*dx),
                assemble(self.p0test*abs(Psi_v)*dx),
                assemble(self.p0test*abs(Psi_eta)*dx),
            ]
        else:
            return [
                sqrt(assemble(self.p0test*Psi_u*Psi_u*dx)),
                sqrt(assemble(self.p0test*Psi_v*Psi_v*dx)),
                sqrt(assemble(self.p0test*Psi_eta*Psi_eta*dx)),
            ]

    def flux_terms(self, uv, elev, uv_old, elev_old):
        """
        Compute flux jump terms over a single
        timestep, given current solution tuple
        `(uv, elev)` and lagged solution tuple
        `(uv_old, elev_old)`.

        If :attr:`timestepper` is set to
        `'SteadyState'` then only the `uv` and
        `elev` arguments are used.
        """
        # TODO: Account for boundary conditions
        if self.steady:
            args = (uv, elev)
            return [
                self._psi_u_steady(*args),
                self._psi_v_steady(*args),
                self._psi_eta_steady(*args),
            ]
        else:
            args = (uv, elev, uv_old, elev_old)
            return [
                self._psi_u_unsteady(*args),
                self._psi_v_unsteady(*args),
                self._psi_eta_unsteady(*args),
            ]

    def recover_laplacians(self, uv, elev):
        """
        Recover the Laplacian of solution
        tuple `(uv, elev)`.
        """
        P1_vec = VectorFunctionSpace(self.mesh, "CG", 1)
        g, phi = TrialFunction(P1_vec), TestFunction(P1_vec)
        a = inner(phi, g)*dx
        sp = {
            'ksp_type': 'gmres',
            'ksp_gmres_restart': 20,
            'ksp_rtol': 1.0e-05,
            'pc_type': 'sor',
        }
        projections = [Function(P1_vec) for i in range(3)]
        for f, proj in zip((uv[0], uv[1], elev), projections):
            L = f*dot(phi, self.n)*ds - div(phi)*f*dx
            solve(a == L, proj, solver_parameters=sp)
        if self.norm_type == 'L1':
            return [interpolate(abs(div(proj)), self.P0) for proj in projections]
        else:
            return [sqrt(interpolate(inner(div(proj), div(proj)), self.P0)) for proj in projections]

    def difference_quotient(self, uv, elev, uv_old, elev_old, uv_adj, elev_adj, uv_adj_next, elev_adj_next):
        """
        Evaluate the dual weighted residual
        error estimator in difference quotient
        formulation.
        """
        Psi = self.strong_residuals(uv, elev, uv_old, elev_old)
        psi = self.flux_terms(uv, elev, uv_old, elev_old)
        R = self.recover_laplacians(uv_adj, elev_adj)
        if not self.steady:
            for R_i, R_old_i in zip(R, self.recover_laplacians(uv_adj_next, elev_adj_next)):
                R_i += R_old_i
                R_i *= 0.5
        dq = Function(self.P0, name="Difference quotient")
        for Psi_i, psi_i, R_i in zip(Psi, psi, R):
            dq.project(dq + (Psi_i + psi_i/sqrt(self.h))*R_i)
        dq.interpolate(abs(dq))  # Ensure positivity
        return dq
