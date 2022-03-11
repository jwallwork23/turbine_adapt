from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
import pyadjoint


__all__ = ["GoalOrientedTidalFarm"]


class GoalOrientedTidalFarm(GoalOrientedMeshSeq):
    """
    Class to facilitate goal-oriented metric-based mesh
    adaptive simulations of tidal farm modelling problems.
    """

    def __init__(self, options, root_dir, num_subintervals, qoi="energy"):
        """
        :arg options: :class:`FarmOptions` encapsulating the problem
        :arg root_dir: directory where metrics should be stored
        """
        self.options = options
        self.root_dir = root_dir
        self.integrated_quantity = qoi
        self.keep_log = False

        # Partition time interval
        dt = options.timestep
        dt_per_export = [int(options.simulation_export_time / dt)] * num_subintervals
        time_partition = TimePartition(
            options.simulation_end_time,
            num_subintervals,
            dt,
            ["swe2d"],
            timesteps_per_export=dt_per_export,
        )

        # Set initial meshes to default
        initial_meshes = [
            Mesh(options.mesh2d) for subinterval in time_partition.subintervals
        ]

        # Create GoalOrientedMeshSeq
        super(GoalOrientedTidalFarm, self).__init__(
            time_partition,
            initial_meshes,
            None,
            None,
            None,
            None,
            qoi_type="time_integrated",
        )

    def get_function_spaces(self, mesh):
        """
        Get the mixed finite element space for a given mesh.
        """
        uv = ("DG", 1)
        if self.options.element_family == "dg-dg":
            elev = ("DG", 1)
        elif self.options.element_family == "dg-cg":
            elev = ("CG", 2)
        else:
            raise NotImplementedError
        return {
            "swe2d": MixedFunctionSpace(
                [
                    VectorFunctionSpace(mesh, *uv, name="U_2d"),
                    get_functionspace(mesh, *elev, name="H_2d"),
                ]
            )
        }

    def get_solver(self):
        options = self.options

        def solver(i, ic, **model_options):
            """
            Solve forward over time window (`t_start`, `t_end`).
            """
            t_start, t_end = self.time_partition.subintervals[i]
            mesh = ic.swe2d.function_space().mesh()
            options.rebuild_mesh_dependent_components(mesh)
            options.simulation_end_time = t_end
            i_export = int(np.round(t_start / options.simulation_export_time))

            # Create a new solver object and assign boundary conditions
            solver_obj = FarmSolver(options, mesh=mesh, keep_log=self.keep_log)
            self.keep_log = True
            options.apply_boundary_conditions(solver_obj)
            compute_power = model_options.pop("compute_power", False)
            compute_vorticity = model_options.pop("compute_vorticity", False)
            model_options.setdefault("no_exports", True)
            options.update(model_options)
            if not options.no_exports:
                options.fields_to_export = ["uv_2d"]

            # Callback which writes power output to HDF5
            if compute_power:
                cb = PowerOutputCallback(solver_obj)
                cb._create_new_file = i_export == 0
                solver_obj.add_callback(cb, "timestep")

            # Callback which writes vorticity to vtk
            if compute_vorticity and not options.no_exports:
                options.fields_to_export.append("vorticity_2d")
                vorticity_2d = Function(
                    solver_obj.function_spaces.P1_2d, name="vorticity_2d"
                )
                uv_2d = solver_obj.fields.uv_2d
                vorticity_calculator = VorticityCalculator2D(uv_2d, vorticity_2d)
                solver_obj.add_new_field(
                    vorticity_2d,
                    "vorticity_2d",
                    "Vorticity",
                    "Vorticity2d",
                    unit="s-1",
                    preproc_func=vorticity_calculator.solve,
                )

            # Set initial conditions for current mesh iteration
            solver_obj.create_exporters()
            uv, elev = ic.swe2d.split()
            solver_obj.assign_initial_conditions(uv=uv, elev=elev)
            solver_obj.i_export = i_export
            solver_obj.next_export_t = i_export * options.simulation_export_time
            solver_obj.iteration = int(
                np.ceil(solver_obj.next_export_t / options.timestep)
            )
            solver_obj.simulation_time = t_start
            solver_obj.export_initial_state = False
            if not options.no_exports:
                solver_obj.exporters["vtk"].set_next_export_ix(i_export)

            # Setup QoI
            qoi = self.get_qoi(i)

            def update_forcings(t):
                options.update_forcings(t)
                self.J += qoi({"swe2d": solver_obj.fields.solution_2d}, t)

            # Solve forward on current subinterval
            solver_obj.iterate(
                update_forcings=update_forcings, export_func=options.export_func
            )
            return AttrDict({"swe2d": solver_obj.fields.solution_2d})

        return solver

    def get_initial_condition(self):
        """
        Near-zero initial velocity and an
        initial elevation which satisfies
        the boundary conditions.
        """
        q = Function(self.function_spaces.swe2d[0])
        u, eta = q.split()
        ramp = self.options.ramp
        assert ramp is not None
        print_output("Initialising with ramped hydrodynamics")
        u_ramp, eta_ramp = ramp.split()
        u.project(u_ramp)
        eta.project(eta_ramp)
        return {"swe2d": q}

    def get_qoi(self, i):
        """
        Currently supported QoIs:

        * 'energy' - energy output of tidal farm.
        """
        if self.integrated_quantity == "energy":
            P0 = get_functionspace(self[i], "DG", 0)
            turbine_drag = Constant(self.options.quadratic_drag_coefficient)
            ct = self.options.corrected_thrust_coefficient * Constant(pi / 8)
            for subdomain_id in self.options.farm_ids:  # TODO: Use union
                subset = self[i].cell_subset(subdomain_id)
                turbine_drag = turbine_drag + interpolate(ct, P0, subset=subset)

            def qoi(sol, t):
                u, eta = sol["swe2d"].split()
                j = assemble(turbine_drag * pow(sqrt(dot(u, u)), 3) * dx)
                if pyadjoint.tape.annotate_tape():
                    j.block_variable.adj_value = 1.0
                return j

        else:
            raise NotImplementedError  # TODO: Consider different QoIs

        return qoi

    def fixed_point_iteration(self, **parsed_args):
        """
        Apply a goal-oriented metric-based mesh adaptation
        fixed point iteration loop for a tidal farm
        modelling problem.
        """
        parsed_args = AttrDict(parsed_args)
        options = self.options
        expected = {
            "miniter",
            "maxiter",
            "load_index",
            "qoi_rtol",
            "element_rtol",
            "error_indicator",
            "approach",
            "h_min",
            "h_max",
            "turbine_h_max",
            "target_complexity",
            "flux_form",
            "norm_order",
        }
        assert expected.issubset(set(parsed_args.keys())), "Missing required arguments"
        output_dir = options.output_directory
        end_time = options.simulation_end_time
        dt = options.timestep
        approach = parsed_args.approach.split("_")[0]
        hmax = Constant(parsed_args.h_max)
        turbine_hmax = Constant(parsed_args.turbine_h_max)
        target = (
            end_time / dt * parsed_args.target_complexity
        )  # Convert to space-time complexity
        num_subintervals = self.num_subintervals
        timesteps = [dt] * num_subintervals

        # Enter fixed point iteration
        miniter = parsed_args.miniter
        maxiter = parsed_args.maxiter
        if miniter > maxiter:
            miniter = maxiter
        qoi_rtol = parsed_args.qoi_rtol
        element_rtol = parsed_args.element_rtol
        converged = False
        converged_reason = None
        num_cells_old = None
        J_old = None
        load_index = parsed_args.load_index
        fp_iteration = load_index
        while fp_iteration <= maxiter:
            outfiles = AttrDict({})
            if fp_iteration < miniter:
                converged = False
            elif fp_iteration == maxiter:
                converged = True
                if converged_reason is None:
                    converged_reason = "maximum number of iterations reached"

            # Load meshes, if requested
            if load_index > 0 and fp_iteration == load_index:
                for i in range(num_subintervals):
                    mesh_fname = os.path.join(output_dir, f"mesh_fp{fp_iteration}_{i}")
                    if os.path.exists(mesh_fname + ".h5"):
                        print_output(
                            f"\n--- Loading plex data for mesh {i+1}\n{mesh_fname}"
                        )
                    else:
                        raise IOError(f"Cannot load mesh file {mesh_fname}.")
                    plex = PETSc.DMPlex().create()
                    plex.createFromFile(mesh_fname + ".h5")
                    self.meshes[i] = Mesh(plex)

            # Create metric Functions
            metrics = [
                Function(TensorFunctionSpace(mesh, "CG", 1), name="Metric")
                for mesh in self.meshes
            ]

            # Load metric data, if available
            loaded = False
            if fp_iteration == load_index:
                for i, metric in enumerate(metrics):
                    if load_index == 0:
                        metric_fname = os.path.join(self.root_dir, f"metric{i}")
                    else:
                        metric_fname = os.path.join(
                            output_dir, f"metric{i}_fp{fp_iteration}"
                        )
                    if os.path.exists(metric_fname + ".h5"):
                        print_output(
                            f"\n--- Loading metric data on mesh {i+1}\n{metric_fname}"
                        )
                        try:
                            with DumbCheckpoint(metric_fname, mode=FILE_READ) as chk:
                                chk.load(metric, name="Metric")
                            loaded = True
                        except Exception:
                            print_output(f"Cannot load metric data on mesh {i+1}")
                            loaded = False
                            break
                    else:
                        assert not loaded, "Only partial metric data available"
                        break

            # Otherwise, solve forward and adjoint
            if not loaded:

                # Solve forward and adjoint on each subinterval
                if converged:
                    with pyadjoint.stop_annotating():
                        print_output("\n--- Final forward run\n")
                        self.get_checkpoints(
                            solver_kwargs=dict(no_exports=False, compute_power=True),
                            run_final_subinterval=True,
                        )
                else:
                    print_output(f"\n--- Forward-adjoint sweep {fp_iteration}\n")
                    solutions = self.solve_adjoint()

                # Check for QoI convergence
                if J_old is not None:
                    if (
                        abs(self.J - J_old) < qoi_rtol * J_old
                        and fp_iteration >= miniter
                    ):
                        converged = True
                        converged_reason = "converged quantity of interest"
                        with pyadjoint.stop_annotating():
                            print_output("\n--- Final forward run\n")
                            self.get_checkpoints(
                                solver_kwargs=dict(
                                    no_exports=False, compute_power=True
                                ),
                                run_final_subinterval=True,
                            )
                J_old = self.J

                # Escape if converged
                if converged:
                    print_output(
                        f"Termination due to {converged_reason} after {fp_iteration+1}"
                        + f" iterations\nEnergy output: {self.J/3.6e+09} MWh"
                    )
                    break

                # Create vtu output files
                outfiles.forward = File(f"{output_dir}/Forward2d.pvd")
                outfiles.forward_old = File(f"{output_dir}/ForwardOld2d.pvd")
                outfiles.adjoint_next = File(f"{output_dir}/AdjointNext2d.pvd")
                outfiles.adjoint = File(f"{output_dir}/Adjoint2d.pvd")

                # Construct metric
                error_indicators = []
                hessians = []
                with pyadjoint.stop_annotating():
                    print_output(f"\n--- Error estimation {fp_iteration}\n")
                    for i, mesh in enumerate(self.meshes):
                        options.rebuild_mesh_dependent_components(mesh)
                        options.get_bnd_conditions(self.function_spaces.swe2d[i])
                        update_forcings = options.update_forcings

                        # Create error estimator
                        ee = ErrorEstimator(
                            options,
                            mesh=mesh,
                            error_estimator=parsed_args.error_indicator,
                        )
                        error_indicators_step = [
                            Function(ee.P0)
                            for field in range(1 if approach == "isotropic" else 3)
                        ]
                        hessians_step = [
                            Function(metrics[i])
                            for field in range(0 if approach == "isotropic" else 3)
                        ]

                        # Loop over all exported timesteps
                        N = len(solutions.swe2d.adjoint[i])
                        for j in range(N):
                            if i < num_subintervals - 1 and j == N - 1:
                                continue

                            # Plot fields
                            args = []
                            for f in outfiles:
                                args.extend(solutions.swe2d[f][i][j].split())
                                args[-2].rename(
                                    "Adjoint velocity" if "adjoint" in f else "Velocity"
                                )
                                args[-1].rename(
                                    "Adjoint elevation"
                                    if "adjoint" in f
                                    else "Elevation"
                                )
                                outfiles[f].write(*args[-2:])

                            # Evaluate error indicator
                            update_forcings(
                                i * end_time / num_subintervals + dt * (j + 1)
                            )
                            if approach == "isotropic":
                                _error_indicators_step = [
                                    ee.error_indicator(
                                        *args, flux_form=parsed_args.flux_form
                                    )
                                ]
                                _hessians_step = []
                            else:
                                _error_indicators_step = ee.strong_residuals(*args[:4])
                                _hessians_step = ee.recover_hessians(*args[6:])
                                for _hessian_next, _hessian in zip(
                                    ee.recover_hessians(*args[4:6]), _hessians_step
                                ):
                                    _hessian += _hessian_next
                                    _hessian *= 0.5

                            # Apply trapezium rule
                            if j in (0, N - 1):
                                for _error_indicator in _error_indicators_step:
                                    _error_indicator *= 0.5
                                for _H_i in _hessians_step:
                                    _H_i *= 0.5
                            for error_indicator, _error_indicator in zip(
                                error_indicators_step, _error_indicators_step
                            ):
                                _error_indicator *= dt
                                error_indicator += _error_indicator
                            for hessian, _hessian in zip(hessians_step, _hessians_step):
                                _hessian *= dt
                                hessian += _hessian
                        error_indicators.append(error_indicators_step)
                        hessians.append(hessians_step)

                    # Plot error indicators
                    if approach == "isotropic":
                        outfiles.error = File(
                            os.path.join(output_dir, "Indicator2d.pvd")
                        )
                        for error_indicator in error_indicators:
                            error_indicator[0].rename("Error indicator")
                            outfiles.error.write(error_indicator[0])

                    # Construct metrics
                    for i, error_indicator in enumerate(error_indicators):
                        if approach == "isotropic":
                            metrics[i].assign(isotropic_metric(error_indicator[0]))
                        else:
                            metrics[i].assign(
                                anisotropic_metric(
                                    error_indicator, hessians[i], element_wise=False
                                )
                            )

                        print_output(f"\n--- Storing metric data on mesh {i+1}\n")
                        metric_fname = os.path.join(
                            output_dir, f"metric{i}_fp{fp_iteration}"
                        )
                        with DumbCheckpoint(metric_fname, mode=FILE_CREATE) as chk:
                            chk.store(metrics[i], name="Metric")
                        if fp_iteration == 0:
                            metric_fname = os.path.join(self.root_dir, f"metric{i}")
                            with DumbCheckpoint(metric_fname, mode=FILE_CREATE) as chk:
                                chk.store(metrics[i], name="Metric")

            # Process metrics
            print_output(f"\n--- Metric processing {fp_iteration}\n")
            metrics = space_time_normalise(
                metrics, end_time, timesteps, target, parsed_args.norm_order
            )

            # Enforce element constraints, accounting for turbines
            h_max = []
            for mesh in self.meshes:
                expr = Constant(hmax)
                P0 = FunctionSpace(mesh, "DG", 0)
                for i, subdomain_id in enumerate(options.farm_ids):  # TODO: Use union
                    subset = mesh.cell_subset(subdomain_id)
                    expr = expr + interpolate(turbine_hmax - hmax, P0, subset=subset)
                hmax_func = interpolate(expr, FunctionSpace(mesh, "CG", 1))
                h_max.append(hmax_func)
            metrics = enforce_element_constraints(metrics, parsed_args.h_min, h_max)

            # Plot metrics
            outfiles.metric = File(os.path.join(output_dir, "Metric2d.pvd"))
            for metric in metrics:
                metric.rename("Metric")
                outfiles.metric.write(metric)

            # Adapt meshes
            print_output(f"\n--- Mesh adaptation {fp_iteration}\n")
            outfiles.mesh = File(os.path.join(output_dir, "Mesh2d.pvd"))
            for i, metric in enumerate(metrics):
                self.meshes[i] = Mesh(adapt(self.meshes[i], metric))
                outfiles.mesh.write(self.meshes[i].coordinates)
            num_cells = [mesh.num_cells() for mesh in self.meshes]

            # Check for convergence of element count
            elements_converged = False
            if num_cells_old is not None:
                elements_converged = True
                for nc, _nc in zip(num_cells, num_cells_old):
                    if abs(nc - _nc) > element_rtol * _nc:
                        elements_converged = False
            num_cells_old = num_cells
            if elements_converged:
                print_output(f"Mesh element count converged to rtol {element_rtol}")
                converged = True
                converged_reason = "converged element counts"

            # Save mesh data to disk
            if COMM_WORLD.size == 1:
                for i, mesh in enumerate(self.meshes):
                    mesh_fname = os.path.join(
                        output_dir, f"mesh_fp{fp_iteration+1}_{i}.h5"
                    )
                    viewer = PETSc.Viewer().createHDF5(mesh_fname, "w")
                    viewer(mesh.topology_dm)

            # Increment
            fp_iteration += 1
        print_output(f"Converged in {fp_iteration+1} iterations due to {converged_reason}")
