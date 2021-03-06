from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
from thetis import *
from thetis.diagnostics import VorticityCalculator2D
from thetis.field_defs import field_metadata
from firedrake.meshadapt import RiemannianMetric, adapt
from firedrake.petsc import PETSc
from firedrake_adjoint import pyadjoint
from pyroteus.utility import File, Mesh
import datetime


__all__ = ["GoalOrientedTidalFarm"]


class GoalOrientedTidalFarm(GoalOrientedMeshSeq):
    """
    Class to facilitate goal-oriented metric-based mesh
    adaptive simulations of tidal farm modelling problems.
    """

    @PETSc.Log.EventDecorator()
    def __init__(self, options, root_dir, num_subintervals, **kwargs):
        """
        :arg options: :class:`FarmOptions` encapsulating the problem
        :arg root_dir: directory where metrics should be stored
        :arg num_subintervals: number of subintervals to use in time
        :kwarg qoi: the integrated quantity of choice
        :kwarg qoi_farm_ids: tags for turbines that contribute towards
            the QoI (defaults to the whole array)
        """
        self.options = options
        self.root_dir = root_dir
        self.integrated_quantity = kwargs.get("qoi", "energy")
        self.keep_log = False
        self.qoi_farm_ids = np.array(kwargs.get("qoi_farm_ids", options.farm_ids))

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

    @PETSc.Log.EventDecorator()
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
                    get_functionspace(mesh, *uv, name="U_2d", vector=True),
                    get_functionspace(mesh, *elev, name="H_2d"),
                ]
            )
        }

    @PETSc.Log.EventDecorator()
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
                if "vorticity_2d" in field_metadata:
                    field_metadata.pop("vorticity_2d")
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
            it = int(np.ceil(solver_obj.next_export_t / options.timestep))
            solver_obj.iteration = it
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

    @PETSc.Log.EventDecorator()
    def get_initial_condition(self):
        """
        Near-zero initial velocity and an
        initial elevation which satisfies
        the boundary conditions.
        """
        fs = self.function_spaces.swe2d[0]
        q = Function(fs)
        u, eta = q.split()
        print_output("Initialising with ramped hydrodynamics")
        u_ramp, eta_ramp = self.options.ramp().split()
        u.project(u_ramp)
        eta.project(eta_ramp)
        return {"swe2d": q}

    @PETSc.Log.EventDecorator()
    def get_qoi(self, i):
        """
        Extract a function for evaluating the quantity of interest (QoI) on
        mesh `i` in the :class:`MeshSeq`.

        Currently, the only supported QoI is the energy output. By default,
        all turbines contribute to the QoI. However, this can be changed by
        altering the `qoi_farm_ids` keyword argument when instantiating the
        :class:`GoalOrientedTidalFarm` object.
        """
        if self.integrated_quantity == "energy":
            ct = self.options.corrected_thrust_coefficient * Constant(pi / 8)
            turbine_drag = Function(get_functionspace(self[i], "DG", 0))
            turbine_drag.assign(self.options.quadratic_drag_coefficient)
            for tag in self.qoi_farm_ids.flatten():
                turbine_drag.assign(ct, subset=self[i].cell_subset(tag))

            def qoi(sol, t):
                u, eta = split(sol["swe2d"])
                j = assemble(turbine_drag * pow(sqrt(dot(u, u)), 3) * dx)
                if pyadjoint.tape.annotate_tape():
                    j.block_variable.adj_value = 1.0
                return j

        else:
            raise NotImplementedError  # TODO: Consider different QoIs

        return qoi

    @PETSc.Log.EventDecorator()
    @pyadjoint.no_annotations
    def _final_run(self):
        print_output("\n--- Final forward run\n")
        pyadjoint.get_working_tape().clear_tape()
        kw = dict(no_exports=False, compute_power=True, compute_vorticity=True)
        self.get_checkpoints(solver_kwargs=kw, run_final_subinterval=True)

    @PETSc.Log.EventDecorator()
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
            "turbine_h_min",
            "turbine_h_max",
            "a_max",
            "target_complexity",
            "base_complexity",
            "flux_form",
            "norm_order",
            "no_final_run",
        }
        if not expected.issubset(set(parsed_args.keys())):
            missing = expected.difference(set(parsed_args.keys()))
            raise AttributeError(f"Missing required arguments {missing}")
        output_dir = options.output_directory
        end_time = options.simulation_end_time
        dt = options.timestep
        approach = parsed_args.approach
        h_min = parsed_args.h_min
        h_max = parsed_args.h_max
        turbine_h_min = parsed_args.turbine_h_min
        turbine_h_max = parsed_args.turbine_h_max
        a_max = parsed_args.a_max
        num_timesteps = int(np.round(end_time / dt))
        target = num_timesteps * parsed_args.target_complexity
        base = num_timesteps * parsed_args.base_complexity
        num_subintervals = self.num_subintervals
        timesteps = [dt] * num_subintervals
        p = parsed_args.norm_order
        no_final_run = parsed_args.no_final_run
        if COMM_WORLD.size > 1:
            raise NotImplementedError("Mmg2d only supports serial mesh adaptation")

        # Enter fixed point iteration
        miniter = parsed_args.miniter
        maxiter = parsed_args.maxiter
        if miniter > maxiter:
            print_output(f"miniter {miniter} and maxiter {maxiter} are incompatible")
            miniter = maxiter
            print_output(f"Setting miniter={miniter}, maxiter={maxiter}")
        qoi_rtol = parsed_args.qoi_rtol
        element_rtol = parsed_args.element_rtol
        converged_reason = None
        load_index = parsed_args.load_index
        if load_index > 0:
            self.keep_log = True
        fp_iteration = load_index
        msg = "Termination due to {:s} after {:d} iterations"
        cells = [] if load_index == 0 else list(np.load(f"{output_dir}/num_cells_progress.npy"))
        qois = [] if load_index == 0 else list(np.load(f"{output_dir}/J_progress.npy"))
        self.J = 0 if load_index == 0 else qois[-1]

        def check_cell_count_convergence():
            if len(cells) >= max(2, miniter):
                elements_converged = True
                for nc, nc_old in zip(cells[-1], cells[-2]):
                    if abs(nc - nc_old) > element_rtol * nc_old:
                        elements_converged = False
                        break
                if elements_converged:
                    return "converged element counts"

        def check_qoi_convergence():
            if len(qois) >= max(2, miniter):
                if abs(qois[-1] - qois[-2]) < qoi_rtol * qois[-2]:
                    return "converged quantity of interest"

        # Check for convergence (of loaded data)
        converged_reason = check_qoi_convergence() or check_cell_count_convergence()

        # Load meshes, if requested
        if load_index > 0:
            for i in range(num_subintervals):
                fname = f"{output_dir}/mesh_fp{fp_iteration}_{i}"
                if not os.path.exists(fname + ".h5"):
                    raise IOError(f"Cannot load mesh file {fname}.")
                print_output(f"\n--- Loading plex {i+1}\n{fname}")
                plex = PETSc.DMPlex().create()
                plex.createFromFile(fname + ".h5")
                self.meshes[i] = Mesh(plex)

        # Do final run, if loaded data has already converged
        if converged_reason is not None:
            if not no_final_run:
                print(f"converged_reason: {converged_reason}")
                self._final_run()
            print_output(msg.format(converged_reason, fp_iteration + 1))
            print_output(f"Energy output: {self.J/3.6e+09} MWh")
            return

        # Enter the fixed point iteration loop
        while fp_iteration <= maxiter:
            print_output(f"Start time for fp_iteration {fp_iteration}: {datetime.datetime.now()}")
            outfiles = AttrDict({})
            if fp_iteration < miniter:
                converged_reason = None
            elif fp_iteration == maxiter:
                converged_reason = converged_reason or "maximum number of iterations reached"

            # Ramp up the target complexity
            target_ramp = ramp_complexity(base, target, fp_iteration)

            # Create metrics
            kw = dict(metric_parameters=dict(dm_plex_metric_verbosity=10))
            metrics = [RiemannianMetric(mesh, **kw) for mesh in self.meshes]
            metric_fns = [metric.function for metric in metrics]

            # Load metric data, if available
            loaded = False
            if fp_iteration == load_index:
                for i, metric in enumerate(metric_fns):
                    fpath = self.root_dir if load_index == 0 else output_dir
                    ext = "" if load_index == 0 else "_fp{fp_iteration}"
                    fname = f"{fpath}/metric{i}{ext}"
                    if os.path.exists(fname + ".h5"):
                        print_output(f"\n--- Loading metric on mesh {i+1}\n{fname}")
                        try:
                            with DumbCheckpoint(fname, mode=FILE_READ) as chk:
                                chk.load(metric, name="Metric")
                            loaded = True
                        except Exception:
                            raise IOError(f"Cannot load metric data on mesh {i+1}")
                    elif loaded:
                        raise IOError("Remove partial metric data")
            if not loaded:

                # Solve forward and adjoint on each subinterval
                if converged_reason is None:
                    print_output(f"\n--- Forward-adjoint sweep {fp_iteration}\n")
                    solutions = self.solve_adjoint()
                else:
                    if not no_final_run:
                        print(f"converged_reason: {converged_reason}")
                        self._final_run()

                # Check for QoI convergence
                converged_reason = converged_reason or check_qoi_convergence()
                if converged_reason is not None:
                    if not no_final_run:
                        print(f"converged_reason: {converged_reason}")
                        self._final_run()
                qois.append(self.J)
                np.save(f"{output_dir}/J_progress.npy", qois)

                # Escape if converged
                if converged_reason is not None:
                    print_output(msg.format(converged_reason, fp_iteration + 1))
                    print_output(f"Energy output: {self.J/3.6e+09} MWh")
                    break

                # Create vtu output files
                outfiles.forward = File(f"{output_dir}/Forward2d.pvd")
                outfiles.forward_old = File(f"{output_dir}/ForwardOld2d.pvd")
                outfiles.adjoint_next = File(f"{output_dir}/AdjointNext2d.pvd")
                outfiles.adjoint = File(f"{output_dir}/Adjoint2d.pvd")

                # Construct metric
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
                            metric=approach,
                            error_estimator=parsed_args.error_indicator,
                        )

                        # Loop over all exported timesteps
                        N = len(solutions.swe2d.adjoint[i])
                        for j in range(N):
                            if i < num_subintervals - 1 and j == N - 1:
                                continue

                            # Plot fields
                            args = []
                            for f in outfiles:
                                args.extend(solutions.swe2d[f][i][j].split())
                                name = "adjoint " if "adjoint" in f else ""
                                args[-2].rename((name + "velocity").capitalize())
                                args[-1].rename((name + "elevation").capitalize())
                                outfiles[f].write(*args[-2:])

                            # Construct metric at current timestep
                            t = i * end_time / num_subintervals + dt * (j + 1)
                            update_forcings(t)
                            metric_step = ee.metric(*args, **parsed_args)

                            # Apply trapezium rule
                            metric_step *= 0.5 * dt if j in (0, N - 1) else dt
                            metric_fns[i] += metric_step

                        # Stash metric data
                        print_output(f"\n--- Storing metric data on mesh {i+1}\n")
                        fname = f"{output_dir}/metric{i}_fp{fp_iteration}"
                        with DumbCheckpoint(fname, mode=FILE_CREATE) as chk:
                            chk.store(metric_fns[i], name="Metric")
                        if fp_iteration == 0:
                            fname = f"{self.root_dir}/metric{i}"
                            with DumbCheckpoint(fname, mode=FILE_CREATE) as chk:
                                chk.store(metric_fns[i], name="Metric")

            # Apply space-time normalisation
            print_output(f"\n--- Metric processing {fp_iteration}\n")
            space_time_normalise(metric_fns, end_time, timesteps, target_ramp, p)

            # Enforce element constraints, accounting for turbines
            hmins = []
            hmaxs = []
            for mesh in self.meshes:
                P0 = get_functionspace(mesh, "DG", 0)
                hmin = Function(P0).assign(h_min)
                hmax = Function(P0).assign(h_max)
                for tag in self.qoi_farm_ids.flatten():
                    hmin.assign(turbine_h_min, subset=mesh.cell_subset(tag))
                    hmax.assign(turbine_h_max, subset=mesh.cell_subset(tag))
                hmins.append(hmin)
                hmaxs.append(hmax)
            enforce_element_constraints(metric_fns, hmins, hmaxs, a_max)

            # Plot metrics
            outfiles.metric = File(f"{output_dir}/Metric2d.pvd")
            for metric in metric_fns:
                outfiles.metric.write(metric)

            # Adapt meshes
            print_output(f"\n--- Mesh adaptation {fp_iteration}\n")
            outfiles.mesh = File(f"{output_dir}/Mesh2d.pvd")
            for i, metric in enumerate(metrics):
                self.meshes[i] = Mesh(adapt(self.meshes[i], metric))
                outfiles.mesh.write(self.meshes[i].coordinates)
            cells.append([mesh.num_cells() for mesh in self.meshes])
            np.save(f"{output_dir}/num_cells_progress.npy", cells)

            # Check for convergence of element count
            check_cell_count_convergence()

            # Save mesh data to disk
            for i, mesh in enumerate(self.meshes):
                fname = f"{output_dir}/mesh_fp{fp_iteration+1}_{i}.h5"
                viewer = PETSc.Viewer().createHDF5(fname, "w")
                viewer(mesh.topology_dm)

            # Increment
            print_output(f"End time for fp_iteration {fp_iteration}: {datetime.datetime.now()}")
            fp_iteration += 1
        print_output(msg.format(converged_reason, fp_iteration + 1))
        print_output(f"Energy output: {self.J/3.6e+09} MWh")
