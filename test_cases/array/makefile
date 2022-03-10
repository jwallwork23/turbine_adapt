all: dir clean mesh plot


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
	python3 meshgen.py 0 aligned
	cd resources && gmsh -2 channel_box_0_aligned.geo
	python3 meshgen.py 1 aligned
	cd resources && gmsh -2 channel_box_1_aligned.geo
	python3 meshgen.py 2 aligned
	cd resources && gmsh -2 channel_box_2_aligned.geo
	python3 meshgen.py 3 aligned
	cd resources && gmsh -2 channel_box_3.geo
	python3 meshgen.py 4 aligned
	cd resources && gmsh -2 channel_box_4.geo
	python3 meshgen.py 5 aligned
	cd resources && gmsh -2 channel_box_5.geo

staggered_mesh:
	python3 meshgen.py 0 staggered
	cd resources && gmsh -2 channel_box_0_staggered.geo
	python3 meshgen.py 1 staggered
	cd resources && gmsh -2 channel_box_1_staggered.geo
	python3 meshgen.py 2 staggered
	cd resources && gmsh -2 channel_box_2_staggered.geo
	python3 meshgen.py 3 staggered
	cd resources && gmsh -2 channel_box_3_staggered.geo
	python3 meshgen.py 4 staggered
	cd resources && gmsh -2 channel_box_4_staggered.geo
	python3 meshgen.py 5 staggered
	cd resources && gmsh -2 channel_box_5_staggered.geo


# --- Plotting

plot_setup:
	for config in aligned staggered; do \
		python3 plot_mesh.py $$config; \
		python3 plot_mesh_reynolds_number.py $$config; \
	done

plot_fixed_mesh:
	for level in 0 1 2 3; do \
		python3 plot_power.py -approach fixed_mesh -level $$level; \
	done
	python3 plot_energy.py -approach fixed_mesh
	python3 plot_vorticity.py -approach fixed_mesh

plot_adapt_vorticity:
	for target in 500000 1000000 2000000 4000000; do \
		python3 plot_power.py -approach vorticity -target $$target; \
	done
	python3 plot_energy.py -approach vorticity
	python3 plot_vorticity.py -approach vorticity


# --- Cleanup

clean:
	cd resources && rm -rf *.msh
