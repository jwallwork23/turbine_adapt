from __future__ import absolute_import
from turbine_adapt.parse import Parser, positive_int
import os
from options import ArrayOptions


# Parse for refinement level
parser = Parser(prog="test_cases/array/meshgen.py")
parser.add_argument(
    "refinement_level",
    help="Number of refinements of farm region",
    type=positive_int,
)
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parsed_args = parser.parse_args()
level = int(parsed_args.refinement_level)
config = parsed_args.configuration

# Boiler plate
code = (
    "//"
    + 80 * "*"
    + f"""
// This geometry file was automatically generated using the `meshgen.py` script
// with refinement level {level:d}.
"""
    + "//"
    + 80 * "*"
    + "\n\n"
)

# Domain and turbine specification
op = ArrayOptions(meshgen=True)
code += "// Domain and turbine specification\n"
code += f"L = {op.domain_length:.0f};\n"
code += f"W = {op.domain_width:.0f};\n"
code += f"D = {op.turbine_diameter:.0f};\n"
code += f"d = {op.turbine_width:.0f};\n"
code += "deltax = 10*D;\ndeltay = 7.5*D;\n"
code += "dx = 100;\n"
dxfarm = [24, 12, 6, 5, 4, 3][level]
code += f"dxfarm = {dxfarm:.1f};\n"
code += f"dxturbine = {min(dxfarm, 6):.1f};\n"

# Channel geometry
code += """
// Channel
Point(1) = {-L/2, -W/2, 0, dx};
Point(2) = {L/2, -W/2, 0, dx};
Point(3) = {L/2, W/2, 0, dx};
Point(4) = {-L/2, W/2, 0, dx};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Physical Line(1) = {1};
Physical Line(2) = {2};
Physical Line(3) = {3};
Physical Line(4) = {4};
Line Loop(1) = {1, 2, 3, 4};
"""

signs = ([-1, 1, 1, -1], [-1, -1, 1, 1])

# Code snippets
point_str = "Point(%d) = {%d*d/2 + %d*deltax, %d*D/2 + %f*deltay, 0, dxturbine};\n"
line_str = "Line(%d) = {%d, %d};\n"
loop_str = "Line Loop(%d) = {%d, %d, %d, %d};\n"

# Turbines
point = 5
line = 5
loop = 2
for col in range(5):
    for row in range(3):
        tag = op.array_ids[row][col]
        code += "\n// turbine %d\n" % (loop - 1)
        for s1, s2 in zip(*signs):
            if config == "aligned":
                code += point_str % (point, s1, col - 2, s2, 1 - row)
            elif config == "staggered":
                code += point_str % (
                    point,
                    s1,
                    col - 2,
                    s2,
                    1 - row + (-1) ** col * 0.25,
                )
            else:
                raise NotImplementedError  # TODO
            point += 1
        for i in range(4):
            code += line_str % (line + i, line + i, line + ((i + 1) % 4))
        code += loop_str % (loop, line, line + 1, line + 2, line + 3)
        line += 4
        loop += 1

# Refined region around turbines
code += "\n// Refined region around the turbines\n"
if config == "aligned":
    point_str = "Point(%d) = {%d*3*deltax, %d*1.3*deltay, 0, dxfarm};\n"
elif config == "staggered":
    point_str = "Point(%d) = {%d*3*deltax, %d*1.55*deltay, 0, dxfarm};\n"
else:
    raise NotImplementedError  # TODO
for s1, s2 in zip(*signs):
    code += point_str % (point, s1, s2)
    point += 1
for i in range(4):
    code += line_str % (line + i, line + i, line + ((i + 1) % 4))
code += loop_str % (loop, line, line + 1, line + 2, line + 3)
line += 4

# Surfaces
code += """
// Surfaces
Plane Surface(1) = {1, 17};
"""
surface_str = "Plane Surface(%d) = {%d};\n"
for surface in range(2, 17):
    code += surface_str % (surface, surface)
code += (
    "Plane Surface(17) = {17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};\n"
)

# Physical surfaces
code += "\n// Physical surfaces\nPhysical Surface(1) = {1, 17};"
surface_str = "\nPhysical Surface(%d) = {%d};"
for surface in range(2, 17):
    code += surface_str % (surface, surface)

# Write to file
with open(
    os.path.join(op.resource_dir, f"channel_box_{level}_{config}.geo"), "w+"
) as f:
    f.write(code)
