import numpy as np
from matplotlib import pyplot as plt
from nurbs.surface import NURBSSurface
from visualization.plotter import NURBSPresenter

def create_bilinear_surface():
    control_points = np.array([
        [[0, 0, 0], [2, 0, 1]],
        [[0, 2, 1], [2, 2, 0]]
    ], dtype=np.float64)
    
    weights = np.array([
        [1.0, 1.0],
        [1.0, 1.0]
    ])
    
    return NURBSSurface(
        control_points=control_points,
        weights=weights,
        knots_u=np.array([0, 0, 1, 1]),
        knots_v=np.array([0, 0, 1, 1]),
        degree_u=1,
        degree_v=1
    )

def run():
    surface = create_bilinear_surface()
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Generate evaluation grid
    u = np.linspace(0, 1, 20)
    v = np.linspace(0, 1, 20)
    U, V = np.meshgrid(u, v)
    
    # Evaluate surface points
    points = np.array([[surface.evaluate(ui, vi) for vi in v] for ui in u])
    
    # Plot surface
    ax.plot_surface(points[:,:,0], points[:,:,1], points[:,:,2], 
                   alpha=0.8, cmap='viridis')
    
    # Plot control points
    ctrl = surface.control_points
    ax.scatter(ctrl[:,:,0], ctrl[:,:,1], ctrl[:,:,2], 
              c='red', s=50, label='Control Points')
    
    ax.legend()
    plt.title("NURBS Surface Demonstration")
    plt.show()

if __name__ == "__main__":
    run()