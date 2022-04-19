all: dir clean mesh plot_setup

APPROACHES		= isotropic_dwr anisotropic_dwr weighted_hessian
CONFIGS			= aligned staggered
FIXED_MESHES	= 0 1 2 3 4
UNIFORM_MESHES	= 0 1
COLUMNS			= {0,1,2,3,4} 2 4
TARGETS			= 10000
MODES			= ramp run both
NP				= 4

.PHONY	:= aligned, staggered, plot_dofs, plot_energy, plot_inflow, plot_power

# --- Setup

dir:
	for approach in fixed_mesh uniform_mesh $(APPROACHES); do \
		for config in $(CONFIGS); do \
			mkdir -p plots/$$config/$$approach; \
			mkdir -p outputs/$$config/$$approach; \
		done; \
	done
	mkdir -p resources

mesh: aligned_mesh staggered_mesh

aligned_mesh:
	for level in $(FIXED_MESHES); do \
		python3 meshgen.py aligned --level $$level; \
	done
	for level in $(UNIFORM_MESHES); do \
		python3 meshgen.py aligned --level $$level --uniform; \
	done
	cd resources/aligned && make

staggered_mesh:
	for level in $(FIXED_MESHES); do \
		python3 meshgen.py staggered --level $$level; \
	done
	for level in $(UNIFORM_MESHES); do \
		python3 meshgen.py staggered --level $$level --uniform; \
	done
	cd resources/staggered && make


# --- Runs

ramp:
	for config in $(CONFIGS); do \
		for level in $(FIXED_MESHES); do \
			mpiexec -np $(NP) python3 ramp.py $$config --level $$level; \
		done; \
	done

longrun:
	for config in $(CONFIGS); do \
		mpiexec -np $(NP) python3 run_fixed_mesh.py $$config --level 0 --ramp_level 0 --num_tidal_cycles 3; \
	done

fixed_mesh:
	for config in $(CONFIGS); do \
		for level in $(FIXED_MESHES); do \
			mpiexec -np $(NP) python3 run_fixed_mesh.py $$config --level $$level --ramp_level $$level; \
		done; \
	done

adaptive:
	for approach in $(APPROACHES); do \
		for config in $(CONFIGS); do \
			for target in $(TARGETS); do \
				python3 run_adapt_goal_based.py $$config -a $$approach --target_complexity $$target; \
			done; \
		done; \
	done


# --- Plotting

plot_setup:
	for config in $(CONFIGS); do \
		python3 plot_mesh.py $$config; \
		python3 plot_mesh_reynolds_number.py $$config; \
	done

plot_power: plot_power_fixed_mesh plot_power_adaptive

plot_power_fixed_mesh:
	for config in $(CONFIGS); do \
		for level in $(FIXED_MESHES); do \
			for mode in $(MODES); do \
				echo "Plotting $$config fixed mesh $$mode level $$level"; \
				python3 plot_power.py $$config $$mode -a fixed_mesh --level $$level; \
			done; \
		done; \
	done

plot_power_adaptive:
	for approach in $(APPROACHES); do \
		for target in $(TARGETS); do \
			for config in $(CONFIGS); do \
				python3 plot_power.py $$config run -a $$approach --target_complexity $$target; \
			done; \
		done; \
	done

# TODO: plot_energy_adaptive

plot_energy_fixed_mesh:
	for config in $(CONFIGS); do \
		for mode in $(MODES); do \
			python3 plot_energy.py $$config $$mode; \
		done; \
	done

plot_inflow:
	for approach in fixed_mesh $(APPROACHES); do \
		python3 $@.py -a $$approach; \
	done

plot_dofs:
	for approach in $(APPROACHES); do \
		for target in $(TARGETS); do \
			python3 $@.py -a $$approach --target_complexity $$target; \
		done; \
	done


# --- Generate plots and data for paper

paper: paper_plots paper_gather paper_power

paper_plots:
	python3 plot_mesh.py aligned
	python3 plot_mesh.py staggered
	python3 plot_inflow.py -a uniform_mesh
	python3 plot_dofs.py -a isotropic_dwr
	python3 plot_dofs.py -a isotropic_dwr --columns
	python3 plot_dofs.py -a anisotropic_dwr
	python3 plot_power.py aligned run -a uniform_mesh --level 0
	python3 plot_power.py staggered run -a uniform_mesh --level 0
	python3 plot_power.py aligned run -a isotropic_dwr --target_complexity 10000
	python3 plot_power.py staggered run -a isotropic_dwr --target_complexity 10000
	python3 plot_power.py aligned run -a anisotropic_dwr --target_complexity 10000
	python3 plot_power.py staggered run -a anisotropic_dwr --target_complexity 10000
	python3 plot_power.py staggered run -a isotropic_dwr --target_complexity 10000 -c 2
	python3 plot_power.py staggered run -a isotropic_dwr --target_complexity 10000 -c 4
	# python3 plot_power.py aligned both -a uniform_mesh --level 0 --num_tidal_cycles 3
	# python3 plot_power.py staggered both -a uniform_mesh --level 0 --num_tidal_cycles 3

paper_gather:
	mkdir -p plots/paper
	cp plots/aligned/aligned_domain.pdf plots/paper/
	cp plots/staggered/staggered_domain.pdf plots/paper/
	cp plots/inflow_velocity_uniform_mesh.pdf plots/paper/
	cp plots/inflow_dynamic_pressure_uniform_mesh.pdf plots/paper/
	cp plots/inflow_power_uniform_mesh.pdf plots/paper/
	cp plots/isotropic_dwr_target10000_dofs.pdf plots/paper/
	cp plots/staggered_isotropic_dwr_target10000_dofs.pdf plots/paper/
	cp plots/anisotropic_dwr_target10000_dofs.pdf plots/paper/
	cp plots/aligned/uniform_mesh/level0/aligned_uniform_mesh_power_output_column_level0_run.pdf plots/paper/
	cp plots/staggered/uniform_mesh/level0/staggered_uniform_mesh_power_output_column_level0_run.pdf plots/paper/
	cp plots/aligned/isotropic_dwr/target10000/aligned_isotropic_dwr_power_output_column_target10000_run.pdf plots/paper/
	cp plots/staggered/isotropic_dwr/target10000/staggered_isotropic_dwr_power_output_column_target10000_run.pdf plots/paper/
	cp plots/aligned/anisotropic_dwr/target10000/aligned_anisotropic_dwr_power_output_column_target10000_run.pdf plots/paper/
	cp plots/staggered/anisotropic_dwr/target10000/staggered_anisotropic_dwr_power_output_column_target10000_run.pdf plots/paper/
	# cp plots/aligned/uniform_mesh/level0/aligned_uniform_mesh_total_power_output_level0_both.pdf plots/paper/
	# cp plots/staggered/uniform_mesh/level0/staggered_uniform_mesh_total_power_output_level0_both.pdf plots/paper/

paper_power:
	python3 compare_power.py aligned -a isotropic_dwr
	python3 compare_power.py aligned -a anisotropic_dwr
	python3 compare_power.py staggered -a isotropic_dwr
	python3 compare_power.py staggered -a anisotropic_dwr
	python3 compare_power.py staggered -a isotropic_dwr -c 2
	python3 compare_power.py staggered -a isotropic_dwr -c 4


# --- Cleanup

clean:
	cd resources && rm -rf *.msh
