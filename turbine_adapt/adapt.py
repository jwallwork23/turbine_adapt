from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
from pyadjoint import stop_annotating


__all__ = ["goal_based_fp"]


def goal_based_fp(solver, initial_condition, qoi, options, root_dir, parsed_args):
    """
    Apply a goal-oriented metric-based mesh adaptation
    fixed point iteration loop for a tidal farm
    modelling problem.

    :arg root_dir: directory where metrics should be stored
    """
    expected = {'miniter', 'maxiter', 'load_index', 'qoi_rtol', 'element_rtol', 'error_indicator', 'approach'
                'h_min', 'h_max', 'turbine_h_max', 'target', 'flux_form', 'num_meshes', 'norm_order'}
    assert issubset(expected, set(parsed_args.keys())), "Missing required arguments"
    output_dir = options.output_directory
    end_time = options.simulation_end_time
    dt = options.timestep
    approach = parsed_args.approach.split('_')[0]
    hmax = Constant(parsed_args.h_max)
    turbine_hmax = Constant(parsed_args.turbine_h_max)
    target = parsed_args.target
    num_meshes = parsed_args.num_meshes
    timesteps = [dt]*num_meshes
    dt_per_export = [int(options.simulation_export_time/dt)]*num_meshes

    # Initial mesh sequence
    meshes = [Mesh(options.mesh2d.coordinates) for i in range(num_meshes)]

    # Enter fixed point iteration
    miniter = parsed_args.miniter
    maxiter = parsed_args.maxiter
    if miniter > maxiter:
        miniter = maxiter
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
                converged_reason = 'maximum number of iterations reached'

        # Load meshes, if requested
        if load_index > 0 and fp_iteration == load_index:
            for i in range(num_meshes):
                mesh_fname = os.path.join(output_dir, f"mesh_fp{fp_iteration}_{i}")
                if not os.path.exists(mesh_fname + '.h5'):
                    raise IOError(f"Cannot load mesh file {mesh_fname}.")
                plex = PETSc.DMPlex().create()
                plex.createFromFile(mesh_fname + '.h5')
                meshes[i] = Mesh(plex)

        # Create function spaces
        element = ("DG", 1) if options.element_family == 'dg-dg' else ("CG", 2)
        spaces = AttrDict({
            'solution_2d': [
                MixedFunctionSpace([
                    VectorFunctionSpace(mesh, "DG", 1, name="U_2d"),
                    get_functionspace(mesh, *element, name="H_2d"),
                ])
                for mesh in meshes
            ]
        })
        metrics = [
            Function(TensorFunctionSpace(mesh, "CG", 1), name="Metric")
            for mesh in meshes
        ]

        # Load metric data for first iteration if available
        loaded = False
        if fp_iteration == load_index:
            for i, metric in enumerate(metrics):
                if load_index == 0:
                    metric_fname = os.path.join(root_dir, f'metric{i}')
                else:
                    metric_fname = os.path.join(output_dir, f'metric{i}_fp{fp_iteration}')
                if os.path.exists(metric_fname + '.h5'):
                    print_output(f"\n--- Loading metric data on mesh {i+1}\n")
                    print_output(metric_fname)
                    loaded = True
                    with DumbCheckpoint(metric_fname, mode=FILE_READ) as chk:
                        chk.load(metric, name="Metric")
                else:
                    assert not loaded, "Only partial metric data available"
                    break

        # Otherwise, solve forward and adjoint
        if not loaded:

            # Solve forward and adjoint on each subinterval
            time_partition = TimePartition(
                end_time, len(spaces['solution_2d']), dt, ['solution_2d'],
                timesteps_per_export=dt_per_export, debug=False,
            )
            args = (solver, initial_condition, qoi, spaces, time_partition)
            if converged:
                with stop_annotating():
                    print_output("\n--- Final forward run\n")
                    J, checkpoints = get_checkpoints(
                        *args, solver_kwargs=dict(no_exports=False, compute_power=True),
                    )
            else:
                print_output(f"\n--- Forward-adjoint sweep {fp_iteration}\n")
                J, solutions = solve_adjoint(*args)

            # Check for QoI convergence
            if J_old is not None:
                if abs(J - J_old) < parsed_args.qoi_rtol*J_old and fp_iteration >= miniter:
                    converged = True
                    converged_reason = 'converged quantity of interest'
                    with stop_annotating():
                        print_output("\n--- Final forward run\n")
                        J, checkpoints = get_checkpoints(
                            *args, solver_kwargs=dict(no_exports=False, compute_power=True),
                        )
            J_old = J

            # Escape if converged
            if converged:
                print_output(f"Termination due to {converged_reason} after {fp_iteration+1} iterations")
                print_output(f"Energy output: {J/3.6e+09} MWh")
                break

            # Create vtu output files
            outfiles.forward = File(os.path.join(output_dir, 'Forward2d.pvd'))
            outfiles.forward_old = File(os.path.join(output_dir, 'ForwardOld2d.pvd'))
            outfiles.adjoint_next = File(os.path.join(output_dir, 'AdjointNext2d.pvd'))
            outfiles.adjoint = File(os.path.join(output_dir, 'Adjoint2d.pvd'))

            # Construct metric
            error_indicators = []
            hessians = []
            with stop_annotating():
                print_output(f"\n--- Error estimation {fp_iteration}\n")
                for i, mesh in enumerate(meshes):
                    options.rebuild_mesh_dependent_components(mesh)
                    options.get_bnd_conditions(spaces.solution_2d[i])
                    update_forcings = options.update_forcings

                    # Create error estimator
                    ee = ErrorEstimator(options, mesh=mesh, error_estimator=parsed_args.error_indicator)
                    error_indicators_step = [
                        Function(ee.P0)
                        for field in range(1 if approach == 'isotropic' else 3)]
                    hessians_step = [
                        Function(metrics[i])
                        for field in range(0 if approach == 'isotropic' else 3)
                    ]

                    # Loop over all exported timesteps
                    N = len(solutions.solution_2d.adjoint[i])
                    for j in range(N):
                        if i < num_meshes-1 and j == N-1:
                            continue

                        # Plot fields
                        args = []
                        for f in outfiles:
                            args.extend(solutions.solution_2d[f][i][j].split())
                            args[-2].rename("Adjoint velocity" if 'adjoint' in f else "Velocity")
                            args[-1].rename("Adjoint elevation" if 'adjoint' in f else "Elevation")
                            outfiles[f].write(*args[-2:])

                        # Evaluate error indicator
                        update_forcings(i*end_time/num_meshes + dt*(j + 1))
                        if approach == 'isotropic':
                            _error_indicators_step = [
                                ee.error_indicator(*args, flux_form=parsed_args.flux_form)
                            ]
                            _hessians_step = []
                        else:
                            _error_indicators_step = ee.strong_residuals(*args[:4])
                            _hessians_step = ee.recover_hessians(*args[6:])
                            for _hessian_next, _hessian in zip(ee.recover_hessians(*args[4:6]), _hessians_step):
                                _hessian += _hessian_next
                                _hessian *= 0.5

                        # Apply trapezium rule
                        if j in (0, N-1):
                            for _error_indicator in _error_indicators_step:
                                _error_indicator *= 0.5
                            for _H_i in _hessians_step:
                                _H_i *= 0.5
                        for error_indicator, _error_indicator in zip(error_indicators_step, _error_indicators_step):
                            _error_indicator *= dt
                            error_indicator += _error_indicator
                        for hessian, _hessian in zip(hessians_step, _hessians_step):
                            _hessian *= dt
                            hessian += _hessian
                    error_indicators.append(error_indicators_step)
                    hessians.append(hessians_step)

                # Plot error indicators
                if approach == 'isotropic':
                    outfiles.error = File(os.path.join(output_dir, 'Indicator2d.pvd'))
                    for error_indicator in error_indicators:
                        error_indicator[0].rename("Error indicator")
                        outfiles.error.write(error_indicator[0])

                # Construct metrics
                for i, error_indicator in enumerate(error_indicators):
                    if approach == 'isotropic':
                        metrics[i].assign(isotropic_metric(error_indicator[0]))
                    else:
                        metrics[i].assign(anisotropic_metric(error_indicator, hessians[i], element_wise=False))

                    print_output(f"\n--- Storing metric data on mesh {i+1}\n")
                    metric_fname = os.path.join(output_dir, f'metric{i}_fp{fp_iteration}')
                    with DumbCheckpoint(metric_fname, mode=FILE_CREATE) as chk:
                        chk.store(metrics[i], name="Metric")
                    if fp_iteration == 0:
                        metric_fname = os.path.join(root_dir, f'metric{i}')
                        with DumbCheckpoint(metric_fname, mode=FILE_CREATE) as chk:
                            chk.store(metrics[i], name="Metric")

        # Process metrics
        print_output(f"\n--- Metric processing {fp_iteration}\n")
        metrics = space_time_normalise(metrics, end_time, timesteps, target, parsed_args.norm_order)

        # Enforce element constraints, accounting for turbines
        h_max = []
        for mesh in meshes:
            expr = Constant(hmax)
            P0 = FunctionSpace(mesh, "DG", 0)
            for i, subdomain_id in enumerate(options.farm_ids):  # TODO: Use union
                subset = mesh.cell_subset(subdomain_id)
                expr = expr + interpolate(turbine_hmax - hmax, P0, subset=subset)
            hmax_func = interpolate(expr, FunctionSpace(mesh, "CG", 1))
            h_max.append(hmax_func)
        metrics = enforce_element_constraints(
            metrics, parsed_args.h_min, h_max
        )

        # Plot metrics
        outfiles.metric = File(os.path.join(output_dir, 'Metric2d.pvd'))
        for metric in metrics:
            metric.rename("Metric")
            outfiles.metric.write(metric)

        # Adapt meshes
        print_output(f"\n--- Mesh adaptation {fp_iteration}\n")
        outfiles.mesh = File(os.path.join(output_dir, 'Mesh2d.pvd'))
        for i, metric in enumerate(metrics):
            meshes[i] = Mesh(adapt(meshes[i], metric).coordinates)
            outfiles.mesh.write(meshes[i].coordinates)
        num_cells = [mesh.num_cells() for mesh in meshes]

        # Check for convergence of element count
        elements_converged = False
        if num_cells_old is not None:
            elements_converged = True
            for nc, _nc in zip(num_cells, num_cells_old):
                if abs(nc - _nc) > parsed_args.element_rtol*_nc:
                    elements_converged = False
        num_cells_old = num_cells
        if elements_converged:
            print_output(f"Mesh element count converged to rtol {parsed_args.element_rtol}")
            converged = True
            converged_reason = 'converged element counts'

        # Save mesh data to disk
        if COMM_WORLD.size == 1:
            for i, mesh in enumerate(meshes):
                mesh_fname = os.path.join(output_dir, f"mesh_fp{fp_iteration}_{i}.h5")
                viewer = PETSc.Viewer().createHDF5(mesh_fname, 'w')
                viewer(mesh.topology_dm)

        # Increment
        fp_iteration += 1

    # Log convergence reason
    with open(os.path.join(output_dir, 'log'), 'a+') as f:
        f.write(f"Converged in {fp_iteration+1} iterations due to {converged_reason}")
