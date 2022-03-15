all: dir clean mesh plot_setup

CONFIGS			= aligned staggered
FIXED_MESHES	= 0 1 2 3 4
TARGETS			=
# TODO: set target complexities

.PHONY	:= plot_energy

# --- Setup

dir:
	for approach in fixed_mesh hessian vorticity isotropic anisotropic; do \
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
			python3 ramp.py $$config --level $$level; \
		done; \
	done

longrun:
	for config in $(CONFIGS); do \
		python3 run_fixed_mesh.py $$config --level 0 --num_tidal_cycles 3; \
	done

fixed_mesh:
	for config in $(CONFIGS); do \
		for level in $(FIXED_MESHES); do \
			python3 run_fixed_mesh.py $$config --level $$level; \
		done; \
	done

isotropic:
	for config in $(CONFIGS); do \
		for level in $(TARGETS); do \
			python3 run_adapt_goal_based.py $$config --target_complexity $$target; \
		done; \
	done


# --- Plotting

plot_setup:
	for config in $(CONFIGS); do \
		python3 plot_mesh.py $$config; \
		python3 plot_mesh_reynolds_number.py $$config; \
	done

plot_fixed_mesh:
	for config in $(CONFIGS); do \
		for level in $(FIXED_MESHES); do \
			for mode in ramp run both; do \
				python3 plot_power.py $$config $$mode -a fixed_mesh --level $$level; \
			done; \
		done; \
	done

plot_energy:
	for config in $(CONFIGS); do \
		for mode in ramp run both; do \
			python3 plot_energy.py $$config $$mode; \
		done; \
	done


# --- Cleanup

clean:
	cd resources && rm -rf *.msh
