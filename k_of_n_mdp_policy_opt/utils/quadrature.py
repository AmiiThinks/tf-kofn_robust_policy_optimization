def midpoint_quadrature(y, x_bounds):
    y_mid = (y[:-1] + y[1:]) / 2.0
    return ((x_bounds[1] - x_bounds[0]) / len(y) * y_mid).sum()
