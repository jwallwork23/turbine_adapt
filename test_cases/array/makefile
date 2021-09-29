all: dir clean mesh plot


# --- Setup

dir:
	@echo "Creating directories..."
	@mkdir -p resources
	@mkdir -p plots/aligned/fixed_mesh
	@mkdir -p plots/staggered/fixed_mesh
	@mkdir -p plots/aligned/vorticity
	@mkdir -p plots/staggered/vorticity
	@mkdir -p plots/aligned/isotropic
	@mkdir -p plots/staggered/isotropic
	@mkdir -p plots/aligned/anisotropic
	@mkdir -p plots/staggered/anisotropic
	@mkdir -p outputs/aligned/fixed_mesh
	@mkdir -p outputs/staggered/fixed_mesh
	@mkdir -p outputs/aligned/vorticity
	@mkdir -p outputs/staggered/vorticity
	@mkdir -p outputs/aligned/isotropic
	@mkdir -p outputs/staggered/isotropic
	@mkdir -p outputs/aligned/anisotropic
	@mkdir -p outputs/staggered/anisotropic

mesh: aligned_mesh staggered_mesh

aligned_mesh:
	@echo "Generating Delauney meshes for aligned turbine array..."
	@mkdir -p resources
	@python3 meshgen.py 0 aligned
	@cd resources && gmsh -2 channel_box_0_aligned.geo
	@python3 meshgen.py 1 aligned
	@cd resources && gmsh -2 channel_box_1_aligned.geo
	@python3 meshgen.py 2 aligned
	@cd resources && gmsh -2 channel_box_2_aligned.geo
	@python3 meshgen.py 3 aligned
	@cd resources && gmsh -2 channel_box_3.geo
	@python3 meshgen.py 4 aligned
	@cd resources && gmsh -2 channel_box_4.geo
	@python3 meshgen.py 5 aligned
	@cd resources && gmsh -2 channel_box_5.geo

staggered_mesh:
	@echo "Generating Delauney meshes for staggered turbine array..."
	@mkdir -p resources
	@python3 meshgen.py 0 staggered
	@cd resources && gmsh -2 channel_box_0_staggered.geo
	@python3 meshgen.py 1 staggered
	@cd resources && gmsh -2 channel_box_1_staggered.geo
	@python3 meshgen.py 2 staggered
	@cd resources && gmsh -2 channel_box_2_staggered.geo
	@python3 meshgen.py 3 staggered
	@cd resources && gmsh -2 channel_box_3_staggered.geo
	@python3 meshgen.py 4 staggered
	@cd resources && gmsh -2 channel_box_4_staggered.geo
	@python3 meshgen.py 5 staggered
	@cd resources && gmsh -2 channel_box_5_staggered.geo


# --- Plotting

plot_setup:
	@echo "Plotting mesh and fields for turbine array problem..."
	@python3 plot_mesh.py aligned
	@python3 plot_mesh.py staggered
	@python3 plot_mesh_reynolds_number.py aligned
	@python3 plot_mesh_reynolds_number.py staggered

plot_fixed_mesh:
	@echo "Plotting fixed mesh outputs..."
	@python3 plot_power.py -approach fixed_mesh -level 0
	@python3 plot_power.py -approach fixed_mesh -level 1
	@python3 plot_power.py -approach fixed_mesh -level 2
	@python3 plot_power.py -approach fixed_mesh -level 3
	@python3 plot_energy.py -approach fixed_mesh
	@python3 plot_vorticity.py -approach fixed_mesh

plot_adapt_vorticity:
	@echo "Plotting isotropic adaptive outputs..."
	@python3 plot_power.py -approach vorticity -target 500000
	@python3 plot_power.py -approach vorticity -target 1000000
	@python3 plot_power.py -approach vorticity -target 2000000
	@python3 plot_power.py -approach vorticity -target 4000000
	@python3 plot_energy.py -approach vorticity
	@python3 plot_vorticity.py -approach vorticity


# --- Cleanup

clean:
	@echo "Cleaning resource directory..."
	@cd resources && rm -rf *.msh
