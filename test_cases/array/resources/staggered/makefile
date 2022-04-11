all: mesh

mesh: box uniform

box:
	gmsh -2 channel_box_0.geo
	gmsh -2 channel_box_1.geo
	gmsh -2 channel_box_2.geo
	gmsh -2 channel_box_3.geo
	gmsh -2 channel_box_4.geo

uniform:
	gmsh -2 -algo pack channel_uniform_0.geo
	gmsh -2 -algo pack channel_uniform_1.geo
