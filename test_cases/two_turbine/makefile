all: base

base: dir clean base_geo base_mesh

box: box_geo box_mesh


dir:
	@echo "Making directories..."
	@mkdir -p resources/meshes
	@mkdir plots

clean:
	@echo "Cleaning directory..."
	@cd resources/meshes && rm -Rf *.geo *.msh


# --- Initial meshes

base_geo:
	@echo "Generating base geometry files..."
	@python3 meshgen.py 0 0
	@python3 meshgen.py 0 1
	@python3 meshgen.py 0 2

box_geo:
	@echo "Generating refined box geometry files..."
	@python3 meshgen.py 0 0 -dx_refined 1
	@python3 meshgen.py 1 0 -dx_refined 1
	@python3 meshgen.py 2 0 -dx_refined 1
	@python3 meshgen.py 3 0 -dx_refined 1
	@python3 meshgen.py 4 0 -dx_refined 1

geo: base_geo
	@echo "Generating other geometry files..."
	@python3 meshgen.py 1 0
	@python3 meshgen.py 1 1
	@python3 meshgen.py 1 2
	@python3 meshgen.py 2 0
	@python3 meshgen.py 2 1
	@python3 meshgen.py 2 2
	@python3 meshgen.py 3 0
	@python3 meshgen.py 3 1
	@python3 meshgen.py 3 2
	@python3 meshgen.py 4 0
	@python3 meshgen.py 4 1
	@python3 meshgen.py 4 2

base_mesh: base_geo
	@echo "Generating base meshes..."
	@gmsh -2 resources/meshes/channel_0_0.geo
	@gmsh -2 resources/meshes/channel_0_1.geo
	@gmsh -2 resources/meshes/channel_0_2.geo

box_mesh: box_geo
	@echo "Generating refined box meshes..."
	@gmsh -2 resources/meshes/channel_refined_0.geo
	@gmsh -2 resources/meshes/channel_refined_1.geo
	@gmsh -2 resources/meshes/channel_refined_2.geo
	@gmsh -2 resources/meshes/channel_refined_3.geo
	@gmsh -2 resources/meshes/channel_refined_4.geo

mesh: geo base_mesh
	@echo "Generating other meshes..."
	@gmsh -2 resources/meshes/channel_1_0.geo
	@gmsh -2 resources/meshes/channel_1_1.geo
	@gmsh -2 resources/meshes/channel_1_2.geo
	@gmsh -2 resources/meshes/channel_2_0.geo
	@gmsh -2 resources/meshes/channel_2_1.geo
	@gmsh -2 resources/meshes/channel_2_2.geo
	@gmsh -2 resources/meshes/channel_3_0.geo
	@gmsh -2 resources/meshes/channel_3_1.geo
	@gmsh -2 resources/meshes/channel_3_2.geo
	@gmsh -2 resources/meshes/channel_4_0.geo
	@gmsh -2 resources/meshes/channel_4_1.geo
	@gmsh -2 resources/meshes/channel_4_2.geo
