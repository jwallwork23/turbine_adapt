all: dir clean mesh plot_setup

APPROACHES		= isotropic_dwr anisotropic_dwr weighted_hessian
CONFIGS			= aligned staggered
FIXED_MESHES	= 0 1 2 3 4
COLUMNS			= {0,1,2,3,4} 2 4
TARGETS			= 10000
MODES			= ramp run both
NP				= 4

.PHONY	:= plot_dofs, plot_energy, plot_inflow, plot_power

# --- Setup

dir:
	for approach in fixed_mesh $(APPROACHES); do \
		for config in $(CONFIGS); do \
			mkdir -p plots/$$config/$$approach; \
			mkdir -p outputs/$$config/$$approach; \
		done; \
	done
	mkdir -p resources

mesh: aligned_mesh staggered_mesh

aligned_mesh:
	for level in $(FIXED_MESHES); do \
		python3 meshgen.py aligned $$level; \
	done
	cd resources && make aligned_mesh

staggered_mesh:
	for level in $(FIXED_MESHES); do \
		python3 meshgen.py staggered $$level; \
	done
	cd resources && make staggered_mesh


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
	for cols in $(COLUMNS); do \
		for approach in $(APPROACHES); do \
			for config in $(CONFIGS); do \
				for target in $(TARGETS); do \
					python3 run_adapt_goal_based.py $$config -a $$approach --target_complexity $$target -c $$cols; \
				done; \
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


# --- Cleanup

clean:
	cd resources && rm -rf *.msh
