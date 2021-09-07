from turbine_adapt import *


__all__ = ["ErrorEstimator"]


class ErrorEstimator(object):
    """
    Error estimation for shallow water tidal
    turbine modelling applications.
    """
    def __init__(self, options, mesh=None, norm_type='L2', error_estimator='difference_quotient'):
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
        assert self.options.swe_timestepper_type in ('CrankNicolson', 'SteadyState')
        self.steady = self.options.swe_timestepper_type == 'SteadyState'
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
        self.error_estimator = error_estimator
        if self.error_estimator != 'difference_quotient':
            raise NotImplementedError  # TODO

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

    def _get_bnd_functions(self, eta_in, uv_in, bnd_in):
        funcs = self.options.bnd_conditions.get(bnd_in)
        if 'elev' in funcs and 'uv' in funcs:
            eta_ext = funcs['elev']
            uv_ext = funcs['uv']
        elif 'elev' in funcs and 'un' in funcs:
            eta_ext = funcs['elev']
            uv_ext = funcs['un']*self.n
        elif 'elev' in funcs:
            eta_ext = funcs['elev']
            uv_ext = uv_in
        elif 'uv' in funcs:
            eta_ext = eta_in
            uv_ext = funcs['uv']
        elif 'un' in funcs:
            eta_ext = eta_in
            uv_ext = funcs['un']*self.n
        else:
            raise Exception(f"Unsupported bnd type: {funcs.keys()}")
        return eta_ext, uv_ext

    def _psi_u_steady(self, *args):
        if len(args) == 2:
            uv, elev = args
            uv_old, elev_old = uv, elev
        elif len(args) == 4:
            uv, elev, uv_old, elev_old = args
        else:
            raise Exception(f"Expected two or four arguments, got {len(args)}.")
        psi = Function(self.P0)
        H = self.options.bathymetry2d + elev
        H_old = self.options.bathymetry2d + elev_old
        g = physical_constants['g_grav']
        flux_terms = 0
        ibp_terms = 0
        if self.eta_is_dg:

            # External pressure gradient
            head_star = avg(elev_old) + sqrt(avg(H)/g)*jump(uv, self.n)
            flux_terms += -head_star*g
            ibp_terms += g*elev*self.n[0]

        # Advection
        flux_terms += avg(uv_old[0])*jump(uv, self.n)
        ibp_terms += uv[0]*dot(uv, self.n)
        uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
        if self.options.use_lax_friedrichs_velocity:
            gamma = 0.5*abs(dot(avg(uv_old), self.n('-')))*uv_lax_friedrichs
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

        # Boundary terms
        bnd_terms = 0
        bnd_conditions = self.options.bnd_conditions
        for bnd_marker in bnd_conditions:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker))
            eta_ext, uv_ext = self._get_bnd_functions(elev, uv, bnd_marker)
            eta_ext_old, uv_ext_old = self._get_bnd_functions(elev_old, uv_old, bnd_marker)

            # External pressure gradient
            un_jump = inner(uv - uv_ext, self.n)
            if funcs is not None:
                eta_rie = 0.5*(elev + eta_ext) + sqrt(H/g)*un_jump
            elif self.eta_is_dg:
                eta_rie = elev + sqrt(H/g)*un_jump
            bnd_terms += -self.p0test*g*eta_rie*self.n[0]*ds_bnd
            if not self.eta_is_dg:
                bnd_terms += self.p0test*g*eta*self.n[0]*ds_bnd

            # Advection
            if funcs is not None:
                eta_jump = elev_old - eta_ext_old
                un_rie = 0.5*inner(uv_old + uv_ext_old, self.n) + sqrt(g/H_old)*eta_jump
                bnd_terms += -self.p0test*un_rie*0.5*(uv_ext[0] + uv[0])*ds_bnd
            elif self.options.use_lax_friedrichs_velocity:
                gamma = 0.5*abs(dot(uv_old, self.n))*uv_lax_friedrichs
                bnd_terms += -gamma*2*dot(uv, self.n)*self.n[0]

            # Viscosity
            if funcs is not None:
                if 'un' in funcs:
                    delta_uv = (dot(uv, self.n) - funcs['un'])*self.n[0]
                else:
                    if uv_ext is uv:
                        continue
                    delta_uv = (uv - uv_ext)[0]
                if self.options.use_grad_div_viscosity_term:
                    stress_jump = 2.0*nu*sym(outer(delta_uv, self.n))
                else:
                    stress_jump = nu*outer(outer(delta_uv, self.n))
                bnd_terms += -self.p0test*sigma*nu*delta_uv*ds_bnd
                # TODO: What about symmetrisation term?

        # Compute flux norm
        mass_term = self.p0test*self.p0trial*dx
        ibp_terms = self._restrict(ibp_terms)*dS
        if self.norm_type == 'L1':
            flux_terms = 2*avg(self.p0test)*abs(flux_terms)*dS
        else:
            flux_terms = 2*avg(self.p0test)*flux_terms*flux_terms*dS
        sp = {'ksp_type': 'preonly', 'pc_type': 'jacobi'}
        solve(mass_term == flux_terms + ibp_terms + bnd_terms, psi, solver_parameters=sp)
        psi.interpolate(abs(psi))
        return sqrt(psi) if self.norm_type == 'L2' else psi

    def _psi_v_steady(self, *args):
        if len(args) == 2:
            uv, elev = args
            uv_old, elev_old = uv, elev
        elif len(args) == 4:
            uv, elev, uv_old, elev_old = args
        else:
            raise Exception(f"Expected two or four arguments, got {len(args)}.")
        psi = Function(self.P0)
        H = self.options.bathymetry2d + elev_old
        g = physical_constants['g_grav']
        flux_terms = 0
        ibp_terms = 0
        if self.eta_is_dg:

            # External pressure gradient
            head_star = avg(elev_old) + sqrt(avg(H)/g)*jump(uv, self.n)
            flux_terms += -head_star*g
            ibp_terms += g*elev*self.n[1]

        # Advection
        flux_terms += avg(uv[1])*jump(uv, self.n)
        ibp_terms += uv[1]*dot(uv, self.n)
        uv_lax_friedrichs = self.options.lax_friedrichs_velocity_scaling_factor
        if self.options.use_lax_friedrichs_velocity:
            gamma = 0.5*abs(dot(avg(uv_old), self.n('-')))*uv_lax_friedrichs
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

        # Boundary terms
        bnd_terms = 0
        bnd_conditions = self.options.bnd_conditions
        for bnd_marker in bnd_conditions:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker))
            eta_ext, uv_ext = self._get_bnd_functions(elev, uv, bnd_marker)
            eta_ext_old, uv_ext_old = self._get_bnd_functions(elev_old, uv_old, bnd_marker)

            # External pressure gradient
            un_jump = inner(uv - uv_ext, self.n)
            if funcs is not None:
                eta_rie = 0.5*(elev + eta_ext) + sqrt(H/g)*un_jump
            elif self.eta_is_dg:
                eta_rie = elev + sqrt(H/g)*un_jump
            bnd_terms += -self.p0test*g*eta_rie*self.n[1]*ds_bnd
            if not self.eta_is_dg:
                bnd_terms += self.p0test*g*eta*self.n[1]*ds_bnd

            # Advection
            if funcs is not None:
                eta_jump = elev_old - eta_ext_old
                un_rie = 0.5*inner(uv_old + uv_ext_old, self.n) + sqrt(g/H)*eta_jump
                bnd_terms += -self.p0test*un_rie*0.5*(uv_ext[1] + uv[1])*ds_bnd
            elif self.options.use_lax_friedrichs_velocity:
                gamma = 0.5*abs(dot(uv_old, self.n))*uv_lax_friedrichs
                bnd_terms += -gamma*2*dot(uv, self.n)*self.n[1]

            # Viscosity
            if funcs is not None:
                if 'un' in funcs:
                    delta_uv = (dot(uv, self.n) - funcs['un'])*self.n[1]
                else:
                    if uv_ext is uv:
                        continue
                    delta_uv = (uv - uv_ext)[1]
                if self.options.use_grad_div_viscosity_term:
                    stress_jump = 2.0*nu*sym(outer(delta_uv, self.n))
                else:
                    stress_jump = nu*outer(outer(delta_uv, self.n))
                bnd_terms += -self.p0test*sigma*nu*delta_uv*ds_bnd
                # TODO: What about symmetrisation term?

        # Compute flux norm
        mass_term = self.p0test*self.p0trial*dx
        ibp_terms = self._restrict(ibp_terms)*dS
        if self.norm_type == 'L1':
            flux_terms = 2*avg(self.p0test)*abs(flux_terms)*dS
        else:
            flux_terms = 2*avg(self.p0test)*flux_terms*flux_terms*dS
        sp = {'ksp_type': 'preonly', 'pc_type': 'jacobi'}
        solve(mass_term == flux_terms + ibp_terms + bnd_terms, psi, solver_parameters=sp)
        psi.interpolate(abs(psi))
        return sqrt(psi) if self.norm_type == 'L2' else psi

    def _psi_eta_steady(self, *args):
        if len(args) == 2:
            uv, elev = args
            uv_old, elev_old = uv, elev
        elif len(args) == 4:
            uv, elev, uv_old, elev_old = args
        else:
            raise Exception(f"Expected two or four arguments, got {len(args)}.")
        psi = Function(self.P0)
        b = self.options.bathymetry2d
        H = b + elev_old
        g = physical_constants['g_grav']
        flux_terms = 0

        # HUDiv
        ibp_terms = inner(H*uv, self.n)
        if self.eta_is_dg:
            un_rie = avg(dot(uv, self.n)) + sqrt(g/avg(H))*jump(elev*self.n, self.n)
            flux_terms += -avg(H)*un_rie
        bnd_terms = 0
        bnd_conditions = self.options.bnd_conditions
        for bnd_marker in bnd_conditions:
            funcs = bnd_conditions.get(bnd_marker)
            ds_bnd = ds(int(bnd_marker))
            eta_ext, uv_ext = self._get_bnd_functions(elev, uv, bnd_marker)
            eta_ext_old, uv_ext_old = self._get_bnd_functions(elev_old, uv_old, bnd_marker)
            if funcs is not None:
                h_av = 0.5*(H + eta_ext_old + b)
                eta_jump = elev - eta_ext
                un_rie = 0.5*inner(uv + uv_ext, self.n) + sqrt(g/h_av)*eta_jump
                un_jump = inner(uv_old - uv_ext_old, self.n)
                eta_rie = 0.5*(elev_old + eta_ext_old) + sqrt(h_av/g)*un_jump
                h_rie = b + eta_rie
                bnd_terms += -self.p0test*h_rie*un_rie*ds_bnd

        # Compute flux norm
        mass_term = self.p0test*self.p0trial*dx
        ibp_terms = self._restrict(ibp_terms)*dS
        if self.norm_type == 'L1':
            flux_terms = 2*avg(self.p0test)*abs(flux_terms)*dS
        else:
            flux_terms = 2*avg(self.p0test)*flux_terms*flux_terms*dS
        sp = {
            "mat_type": "matfree",
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
        }
        solve(mass_term == flux_terms + ibp_terms + bnd_terms, psi, solver_parameters=sp)
        psi.interpolate(abs(psi))
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
        f = self._psi_u_steady(uv, elev, uv, elev)    # NOTE: Not semi-implicit
        f_old = self._psi_u_steady(uv_old, elev_old, uv_old, elev_old)
        return self.theta*f + (1-self.theta)*f_old

    def _psi_v_unsteady(self, uv, elev, uv_old, elev_old):
        f = self._psi_v_steady(uv, elev, uv, elev)    # NOTE: Not semi-implicit
        f_old = self._psi_v_steady(uv_old, elev_old, uv_old, elev_old)
        return self.theta*f + (1-self.theta)*f_old

    def _psi_eta_unsteady(self, uv, elev, uv_old, elev_old):
        f = self._psi_eta_steady(uv, elev, uv, elev)  # NOTE: Not semi-implicit
        f_old = self._psi_eta_steady(uv_old, elev_old, uv_old, elev_old)
        return self.theta*f + (1-self.theta)*f_old

    def strong_residuals(self, *args):
        """
        Compute the strong residual over a single
        timestep, given current solution tuple
        `(uv, elev)` and lagged solution tuple
        `(uv_old, elev_old)`.

        If :attr:`timestepper` is set to
        `'SteadyState'` then only two `uv` and
        `elev` arguments are used.
        """
        if self.steady:
            Psi_u = self._Psi_u_steady(*args)
            Psi_v = self._Psi_v_steady(*args)
            Psi_eta = self._Psi_eta_steady(*args)
        else:
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
                sqrt(abs(assemble(self.p0test*Psi_u*Psi_u*dx))),
                sqrt(abs(assemble(self.p0test*Psi_v*Psi_v*dx))),
                sqrt(abs(assemble(self.p0test*Psi_eta*Psi_eta*dx))),
            ]

    def flux_terms(self, *args):
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

    def recover_hessians(self, uv, elev):
        """
        Recover the Hessian of solution
        tuple `(uv, elev)`.
        """
        return [
            recover_hessian(uv[0], mesh=self.mesh),
            recover_hessian(uv[1], mesh=self.mesh),
            recover_hessian(elev, mesh=self.mesh),
        ]

    def difference_quotient(self, *args, flux_form=False):
        """
        Evaluate the dual weighted residual
        error estimator in difference quotient
        formulation.
        """
        nargs = len(args)
        assert nargs == 4 if self.steady else 8

        # Terms for standard a posteriori error estimate
        Psi = self.strong_residuals(*args[:nargs//2])
        psi = self.flux_terms(*args[:nargs//2])

        # Weighting term for the adjoint
        if flux_form:
            R = self.flux_terms(*args[nargs//2:])
        else:
            R = self.recover_laplacians(*args[nargs//2:2+nargs//2])
            if not self.steady:  # Average recovered Laplacians
                for R_i, R_old_i in zip(R, self.recover_laplacians(*args[-2:])):
                    R_i += R_old_i
                    R_i *= 0.5

        # Combine the two
        dq = Function(self.P0, name="Difference quotient")
        for Psi_i, psi_i, R_i in zip(Psi, psi, R):
            dq.project(dq + (Psi_i + psi_i/sqrt(self.h))*R_i)
        dq.interpolate(abs(dq))  # Ensure positivity
        return dq

    def error_indicator(self, *args, **kwargs):
        if self.error_estimator == 'difference_quotient':
            flux_form = kwargs.get('flux_form', False)
            return self.difference_quotient(*args, flux_form=flux_form)
        else:
            raise NotImplementedError  # TODO
