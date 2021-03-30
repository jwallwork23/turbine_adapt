all: dir clean mesh plot


# --- Setup

dir:
	@echo "Creating directories..."
	@mkdir -p resources
	@mkdir -p plots/fixed_mesh
	@mkdir -p plots/vorticity
	@mkdir -p plots/dwr
	@mkdir -p outputs/fixed_mesh
	@mkdir -p outputs/vorticity
	@mkdir -p outputs/dwr

mesh:
	@echo "Generating Delauney meshes of turbine array domain..."
	@mkdir -p resources
	@python3 meshgen.py 0
	@cd resources && gmsh -2 channel_box_0.geo
	@python3 meshgen.py 1
	@cd resources && gmsh -2 channel_box_1.geo
	@python3 meshgen.py 2
	@cd resources && gmsh -2 channel_box_2.geo
	@python3 meshgen.py 3
	@cd resources && gmsh -2 channel_box_3.geo
	@python3 meshgen.py 4
	@cd resources && gmsh -2 channel_box_4.geo
	@python3 meshgen.py 5
	@cd resources && gmsh -2 channel_box_5.geo


# --- Runs

fixed_mesh:
	@echo "Running fixed mesh simulations..."
	@python3 run_fixed_mesh.py -level 0
	@python3 run_fixed_mesh.py -level 1
	@python3 run_fixed_mesh.py -level 2
	@python3 run_fixed_mesh.py -level 3

adapt_on_vorticity:
	@echo "Running adaptive simulations based on vorticity..."
	@python3 run_adapt_vorticity.py -target 500000
	@python3 run_adapt_vorticity.py -target 1000000
	@python3 run_adapt_vorticity.py -target 2000000
	@python3 run_adapt_vorticity.py -target 4000000

adapt_on_velocity_hessian:
	@echo "Running adaptive simulations based on velocity Hessian..."
	@python3 run_adapt_hessian.py -target 500000
	@python3 run_adapt_hessian.py -target 1000000
	@python3 run_adapt_hessian.py -target 2000000
	@python3 run_adapt_hessian.py -target 4000000


# --- Plotting

plot_setup:
	@echo "Plotting mesh and fields for turbine array problem..."
	@python3 plot_mesh.py
	@python3 plot_mesh_reynolds_number.py

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
