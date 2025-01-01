def scale_to_sum(A, S):
    k = S / sum(A)
    scaled = [k * a for a in A]
    floor_scaled = [int(s) for s in scaled]
    sum_floor = sum(floor_scaled)
    D = S - sum_floor
    residuals = [s - floor_s for s, floor_s in zip(scaled, floor_scaled)]
    indices = sorted(range(len(A)), key=lambda i: -residuals[i])
    for i in indices[:D]:
        floor_scaled[i] += 1
    return floor_scaled