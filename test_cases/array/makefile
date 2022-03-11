all: dir clean mesh plot

FIXED_MESHES	= 0 1 2 3 4

# --- Setup

dir:
	for approach in fixed_mesh hessian vorticity isotropic anisotropic; do \
		for config in aligned staggered; do \
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

ramp: ramp_aligned ramp_staggered

ramp_aligned:
	python3 ramp.py aligned --level 4

ramp_staggered:
	python3 ramp.py staggered --level 4

fixed: fixed_aligned fixed_staggered

fixed_aligned:
	for level in $(FIXED_MESHES); do \
		python3 run_fixed_mesh.py aligned --level $$level; \
	done

fixed_staggered:
	for level in $(FIXED_MESHES); do \
		python3 run_fixed_mesh.py staggered --level $$level; \
	done


# --- Plotting

plot_setup:
	for config in aligned staggered; do \
		python3 plot_mesh.py $$config; \
		python3 plot_mesh_reynolds_number.py $$config; \
	done

plot_fixed_mesh:
	for level in $(FIXED_MESHES); do \
		python3 plot_power.py -approach fixed_mesh -level $$level; \
	done


# --- Cleanup

clean:
	cd resources && rm -rf *.msh
