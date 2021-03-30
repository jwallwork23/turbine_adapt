//-------------------------------------------
//  A single tidal turbine, centrally
//  positioned in a channel experiencing
//  uniform laminar flow. The square turbine
//  footprint is rotated by 45 degrees so
//  that it does not align with the flow.
//-------------------------------------------
// Domain and turbine specification
L = 19000;
W = 7000;
D = 500;
dx_outer = 100;
dx_inner = 50;
xt0 = 4000;
yt0 = W/2;

// Domain and turbine footprints
Point(1) = {0, 0, 0, dx_outer};
Point(2) = {L, 0, 0, dx_outer};
Point(3) = {L, W, 0, dx_outer};
Point(4) = {0, W, 0, dx_outer};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Physical Line(1) = {4};   // Left boundary
Physical Line(2) = {2};   // Right boundary
Physical Line(3) = {1,3}; // Sides
Line Loop(1) = {1, 2, 3, 4};  // outside loop
Point(5) = {xt0-D*0.70711, yt0, 0, dx_inner};
Point(6) = {xt0, yt0-D*0.70711, 0, dx_inner};
Point(7) = {xt0+D*0.70711, yt0, 0, dx_inner};
Point(8) = {xt0, yt0+D*0.70711, 0, dx_inner};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line Loop(2) = {5, 6, 7, 8};

// Surfaces
Plane Surface(1) = {1, 2};
Plane Surface(2) = {2};
Physical Surface(1) = {1};  // outside turbine
Physical Surface(2) = {2};  // inside turbine 1
