import numpy as np
from numba import njit, int32, float32, uint8

def a_star_pathfind(thickness, start, end,
                    dims=[1.0,1.0,1.0],
                    alpha=0.75,
                    margin=10):
    """
    thickness : 3D numpy array (D × H × W)
    start,end : (z,y,x) tuples
    dims      : [dz, dy, dx]
    alpha     : 0→pure Euclid | 1→pure thickness-weighted
    margin    : how many extra voxels around the bbox of start/end
    """
    D, H, W = thickness.shape
    sz, sy, sx = start
    ez, ey, ex = end

    # 1) bounding box
    z0 = max(min(sz, ez) - margin, 0)
    z1 = min(max(sz, ez) + margin + 1, D)
    y0 = max(min(sy, ey) - margin, 0)
    y1 = min(max(sy, ey) + margin + 1, H)
    x0 = max(min(sx, ex) - margin, 0)
    x1 = min(max(sx, ex) + margin + 1, W)

    sub = thickness[z0:z1, y0:y1, x0:x1]
    dD, dH, dW = sub.shape

    # quick validity
    if not (0 <= sz < D and 0 <= ez < D and
            0 <= sy < H and 0 <= ey < H and
            0 <= sx < W and 0 <= ex < W):
        return []
    if thickness[sz,sy,sx] <= 0 or thickness[ez,ey,ex] <= 0:
        return []

    # 2) flatten + inv_thickness
    flat = sub.ravel().astype(np.float32)
    inv_flat = np.empty_like(flat)
    # inv_flat[i] = 1/flat[i] if flat[i]>0 else 0
    np.divide(flat, 1.0, out=inv_flat, where=(flat>0))
    inv_flat[flat <= 0] = 0.0

    # 3) strides, neighbor offsets, phys dists
    stride_z = dH * dW
    stride_y = dW
    stride_x = 1
    offsets = np.array(( stride_z, -stride_z,
                         stride_y, -stride_y,
                         stride_x, -stride_x),
                       dtype=np.int32)
    neighbor_phys = np.array((dims[0], dims[0],
                              dims[1], dims[1],
                              dims[2], dims[2]),
                             dtype=np.float32)

    # map start/end into sub-volume flat indices
    start_idx = ((sz - z0) * stride_z +
                 (sy - y0) * stride_y +
                 (sx - x0) * stride_x)
    end_idx   = ((ez - z0) * stride_z +
                 (ey - y0) * stride_y +
                 (ex - x0) * stride_x)

    # 4) run the JIT-compiled core
    alpha32   = np.float32(alpha)
    flat_path = _astar_jit(inv_flat, offsets, neighbor_phys,
                           start_idx, end_idx,
                           dD, dH, dW, alpha32)
    if flat_path.size == 0:
        return []

    # 5) unpack back to global (z,y,x)
    out = []
    for idx in flat_path:
        z = idx // (dH * dW) + z0
        rem = idx %  (dH * dW)
        y = rem // dW + y0
        x = rem %  dW + x0
        out.append((int(z), int(y), int(x)))
    return out


@njit(
    (float32[:], int32[:], float32[:],
     int32, int32, int32, int32, int32, float32),
    nogil=True
)
def _astar_jit(inv_flat, offsets, neighbor_phys,
               start_idx, end_idx,
               dD, dH, dW, alpha):
    """
    inv_flat      : 1D 1/thickness (0 if blocked)
    offsets       : 6 neighbor-index offsets
    neighbor_phys : 6 physical distances for those offsets
    start_idx, end_idx : flattened indices
    dD, dH, dW    : subvolume shape
    alpha         : interpolation factor
    """
    N = inv_flat.shape[0]

    # --- manual min-heap arrays ---
    maxsize = N + 1
    heap_f   = np.empty(maxsize, dtype=np.float32)
    heap_idx = np.empty(maxsize, dtype=np.int32)
    heap_n   = 0

    def heap_push(fval, vidx):
        nonlocal heap_n
        heap_n += 1
        i = heap_n
        heap_f[i]   = fval
        heap_idx[i] = vidx
        # bubble up
        while i > 1:
            p = i >> 1
            if heap_f[p] <= heap_f[i]:
                break
            # swap
            tf = heap_f[p]; ti = heap_idx[p]
            heap_f[p], heap_idx[p] = heap_f[i], heap_idx[i]
            heap_f[i], heap_idx[i] = tf, ti
            i = p

    def heap_pop():
        nonlocal heap_n
        if heap_n == 0:
            return -1, 0.0
        root_f   = heap_f[1]
        root_idx = heap_idx[1]
        heap_f[1], heap_idx[1] = heap_f[heap_n], heap_idx[heap_n]
        heap_n -= 1
        # bubble down
        i = 1
        while True:
            l = i << 1
            r = l + 1
            smallest = i
            if l <= heap_n and heap_f[l] < heap_f[smallest]:
                smallest = l
            if r <= heap_n and heap_f[r] < heap_f[smallest]:
                smallest = r
            if smallest == i:
                break
            tf = heap_f[i]; ti = heap_idx[i]
            heap_f[i], heap_idx[i] = heap_f[smallest], heap_idx[smallest]
            heap_f[smallest], heap_idx[smallest] = tf, ti
            i = smallest
        return root_idx, root_f

    # g-scores, parent, closed mask
    g      = np.full(N, 1e30, dtype=np.float32)
    parent = np.full(N,   -1, dtype=np.int32)
    closed = np.zeros(N,   dtype=uint8)

    # unpack end coords once
    ez = end_idx // (dH * dW)
    rem= end_idx %  (dH * dW)
    ey = rem    //  dW
    ex = rem    %   dW

    # seed start
    g[start_idx] = 0.0
    # heuristic(start) = manh_phys * [(1-alpha) + alpha*invt]
    sz = start_idx // (dH*dW)
    rem= start_idx %  (dH*dW)
    sy = rem    //  dW
    sx = rem    %   dW
    manh0 = (abs(sz-ez)*neighbor_phys[0] +
             abs(sy-ey)*neighbor_phys[2] +
             abs(sx-ex)*neighbor_phys[4])
    hf0 = manh0 * ((1.0 - alpha) + alpha * inv_flat[start_idx])
    heap_push(hf0, start_idx)

    # main A* loop
    while heap_n > 0:
        current, _f = heap_pop()
        if closed[current]:
            continue
        if current == end_idx:
            # reconstruct path
            length = 0
            tmp = current
            while tmp >= 0:
                length += 1
                tmp = parent[tmp]
            path = np.empty(length, dtype=np.int32)
            i = length - 1
            tmp = current
            while tmp >= 0:
                path[i] = tmp
                i -= 1
                tmp = parent[tmp]
            return path

        closed[current] = 1

        cz = current // (dH*dW)
        rem= current %  (dH*dW)
        cy = rem     //  dW
        cx = rem     %  dW

        for k in range(6):
            nbr = current + offsets[k]
            # out of bounds or blocked?
            if nbr < 0 or nbr >= N:
                continue
            # prevent wrapping on x-moves
            if k == 4 and cx == (dW-1): continue
            if k == 5 and cx == 0:       continue

            invt = inv_flat[nbr]
            if invt == 0.0:
                continue

            # cost to move → phys_dist * (1/thickness)
            tg = g[current] + neighbor_phys[k] * invt
            if tg < g[nbr]:
                g[nbr]      = tg
                parent[nbr] = current

                # heuristic at nbr
                nz = nbr // (dH*dW)
                rem2 = nbr % (dH*dW)
                ny = rem2 // dW
                nx = rem2 %  dW
                manh2 = (abs(nz-ez)*neighbor_phys[0] +
                         abs(ny-ey)*neighbor_phys[2] +
                         abs(nx-ex)*neighbor_phys[4])
                hf = tg + manh2 * ((1.0 - alpha) + alpha * invt)
                heap_push(hf, nbr)

    # unreachable
    return np.empty(0, dtype=np.int32)
