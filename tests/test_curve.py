import numpy as np
from nurbs.curve import NURBSCurve


class TestNURBSCurve:
    def test_circle_approximation(self):
        # Quarter circle approximation
        ctrl = np.array([
            [1, 0],
            [1, 1],
            [0, 1]
        ], dtype=np.float64)
        weights = np.array([1, np.sqrt(2)/2, 1])
        knots = np.array([0, 0, 0, 1, 1, 1], dtype=np.float64)

        nurbs = NURBSCurve(ctrl, weights, knots, 2)
        point = nurbs.evaluate(0.5)

        assert np.allclose(point, [np.sqrt(2)/2, np.sqrt(2)/2], atol=1e-3)

    def test_linear_curve(self):
        ctrl = np.array([[0, 0], [1, 1]], dtype=np.float64)
        knots = np.array([0, 0, 1, 1], dtype=np.float64)
        nurbs = NURBSCurve(ctrl, np.ones(2), knots, 1)

        assert np.allclose(nurbs.evaluate(0.5), [0.5, 0.5])

    def test_clamped_uniform_knots(self):
        """Test standard clamped uniform knot vector"""
        knots = np.array([0,0,0,1,2,3,3,3], dtype=np.float64)  # Degree 2
        control_points = np.random.rand(5, 3)  # 5 points in 3D
        nurbs = NURBSCurve(control_points, np.ones(5), knots, degree=2)
        
        # Verify boundary conditions
        assert nurbs.find_span(0.0) == 2  # First internal span
        assert nurbs.find_span(3.0) == 4  # Last internal span
        
        # Verify intermediate spans
        assert nurbs.find_span(0.5) == 2  # [0,1)
        assert nurbs.find_span(1.5) == 3  # [1,2)
        assert nurbs.find_span(2.5) == 4  # [2,3)

    # def test_periodic_knots(self):
    #     """Test non-clamped periodic knot vector"""
    #     knots = np.array([0,1,2,3,4,5,6,7], dtype=np.float64)  # Degree 2
    #     control_points = np.random.rand(5, 2)
    #     nurbs = NURBSCurve(control_points, np.ones(5), knots, degree=2)
        
    #     # Verify full range search
    #     assert nurbs.find_span(2.5) == 2  # [2,3)
    #     assert nurbs.find_span(5.5) == 4  # [5,6)
    #     assert nurbs.find_span(6.5) == 5  # [6,7)

    def test_edge_cases(self):
        """Test exact knot values"""
        knots = np.array([0,0,0,1,1,1], dtype=np.float64)  # Degree 2
        control_points = np.random.rand(3, 3)
        nurbs = NURBSCurve(control_points, np.ones(3), knots, degree=2)
        
        assert nurbs.find_span(0.0) == 2  # Start of valid range
        assert nurbs.find_span(1.0) == 2  # End of valid range
        assert nurbs.find_span(0.999) == 2  # Near upper limit
        assert nurbs.find_span(1.001) == 2  # Beyond valid range

    # def test_high_degree(self):
    #     """Test cubic B-spline with 10 control points"""
    #     knots = np.array([0]*4 + [1,2,3,4] + [5]*4, dtype=np.float64)  # Degree 3
    #     control_points = np.random.rand(10, 2)
    #     nurbs = NURBSCurve(control_points, np.ones(10), knots, degree=3)
        
    #     assert nurbs.find_span(1.5) == 3  # [1,2)
    #     assert nurbs.find_span(3.5) == 5  # [3,4)
    #     assert nurbs.find_span(4.9) == 6  # [4,5)

    def test_linear_basis(self):
        """Test basis functions for degree 1 (linear)"""
        knots = np.array([0.0, 0.0, 1.0, 1.0])
        control_points = np.array([[0,0], [1,1]])
        nurbs = NURBSCurve(control_points, np.ones(2), knots, degree=1)
        
        # At mid-point of linear segment
        basis = nurbs.basis_functions(span=1, t=0.5)
        assert np.allclose(basis, [0.5, 0.5], atol=1e-6)
        
        # Near start
        basis = nurbs.basis_functions(span=1, t=0.1)
        assert np.allclose(basis, [0.9, 0.1], atol=1e-6)

    def test_quadratic_basis(self):
        """Test basis functions for degree 2 (quadratic)"""
        knots = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        control_points = np.array([[0,0], [1,0], [1,1]])
        nurbs = NURBSCurve(control_points, np.ones(3), knots, degree=2)
        
        # Middle of quadratic curve
        basis = nurbs.basis_functions(span=2, t=0.5)
        assert np.allclose(basis, [0.25, 0.5, 0.25], atol=1e-6)
        
        # At 3/4 point
        basis = nurbs.basis_functions(span=2, t=0.75)
        assert np.allclose(basis, [0.0625, 0.375, 0.5625], atol=1e-6)

    def test_cubic_basis_with_multiplicity(self):
        """Test basis with knot multiplicity"""
        knots = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        control_points = np.random.rand(4, 3)
        nurbs = NURBSCurve(control_points, np.ones(4), knots, degree=3)
        
        # Exactly at start point
        basis = nurbs.basis_functions(span=3, t=0.0)
        assert np.allclose(basis, [1.0, 0.0, 0.0, 0.0], atol=1e-6)
        
        # Just before end
        basis = nurbs.basis_functions(span=3, t=0.999)
        assert np.allclose(basis.sum(), 1.0, atol=1e-6)