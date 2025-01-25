import numpy as np
from nurbs.surface import NURBSSurface

class TestNURBSSurface:
    def test_flat_plane(self):
        """Test a simple bilinear surface (flat plane)"""
        # 2x2 control grid forming a unit square at z=0
        control_points = np.array([
            [[0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [1, 1, 0]]
        ], dtype=np.float64)
        
        weights = np.ones((2, 2))
        surface = NURBSSurface(
            control_points=control_points,
            weights=weights,
            knots_u=np.array([0, 0, 1, 1]),
            knots_v=np.array([0, 0, 1, 1]),
            degree_u=1,
            degree_v=1
        )

        # Test corner points
        assert np.allclose(surface.evaluate(0, 0), [0, 0, 0])
        assert np.allclose(surface.evaluate(1, 1), [1, 1, 0])
        
        # Test center point
        center = surface.evaluate(0.5, 0.5)
        assert np.allclose(center, [0.5, 0.5, 0])

    def test_basis_functions(self):
        """Verify basis function properties at known parameter values"""
        # Cubic surface setup
        surface = NURBSSurface(
            control_points=np.random.rand(4, 4, 3),
            weights=np.ones((4, 4)),
            knots_u=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            knots_v=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            degree_u=3,
            degree_v=3
        )

        # At u=0.0 (full multiplicity)
        basis_u = surface._basis_functions(0.0, 3, surface.knots_u, 3)
        assert np.allclose(basis_u, [1, 0, 0, 0])
        
        # At v=1.0 (full multiplicity)
        basis_v = surface._basis_functions(1.0, 3, surface.knots_v, 3)
        assert np.allclose(basis_v, [0, 0, 0, 1])

    def test_find_span(self):
        """Test knot span localization for both directions"""
        knots_u = np.array([0, 0, 0, 1, 2, 3, 3, 3])
        knots_v = np.array([0, 0, 1, 2, 3, 4, 4])
        
        surface = NURBSSurface(
            control_points=np.random.rand(5, 4, 3),
            weights=np.ones((5, 4)),
            knots_u=knots_u,
            knots_v=knots_v,
            degree_u=2,
            degree_v=1
        )

        # Test u-direction spans
        assert surface._find_span(0.5, knots_u, 2) == 2  # [0,1) -> span 2
        assert surface._find_span(2.5, knots_u, 2) == 4  # [2,3) -> span 4
        
        # Test v-direction spans
        assert surface._find_span(1.5, knots_v, 1) == 2  # [1,2) -> span 2
        assert surface._find_span(3.5, knots_v, 1) == 4  # [3,4) -> span 4

    def test_weighted_surface(self):
        """Test surface deformation through weights"""
        # Control points form a pyramid shape
        control_points = np.array([
            [[-1, -1, 0], [-1, 1, 0]],
            [[1, -1, 0], [1, 1, 0]]
        ], dtype=np.float64)
        
        # Increase weight at center control point (index 1,1)
        weights = np.array([
            [1, 1],
            [1, 5]
        ])
        
        surface = NURBSSurface(
            control_points=control_points,
            weights=weights,
            knots_u=np.array([0, 0, 1, 1]),
            knots_v=np.array([0, 0, 1, 1]),
            degree_u=1,
            degree_v=1
        )

        # Evaluate at weighted point (u=1, v=1)
        point = surface.evaluate(1, 1)
        # Higher weight should "pull" the surface toward [1,1,0]
        assert np.allclose(point, [1, 1, 0], atol=0.1)