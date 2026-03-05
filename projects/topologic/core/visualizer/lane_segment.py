import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── Colour palette (LaneSegNet convention, RGB floats in [0, 1]) ──────────
COLOR_DICT = {
    'centerline':    np.array([243,  90,   2]) / 255,
    'laneline':      np.array([  0,  32, 127]) / 255,
    'ped_crossing':  np.array([255, 192,   0]) / 255,
    'road_boundary': np.array([220,  30,   0]) / 255,
    'topology':      np.array([  0,  32, 127]) / 255,
}

BEV_RANGE = [-50, 50, -25, 25]


# ── Primitive draw helpers ────────────────────────────────────────────────

def _draw_centerline(ax, lane_centerline):
    """Draw a single centerline with direction arrow (LaneSegNet style)."""
    points = np.asarray(lane_centerline['points'])
    color = COLOR_DICT['centerline']
    # polyline
    ax.plot(points[:, 1], points[:, 0], color=color, alpha=1.0, linewidth=0.6)
    # start / end vertices
    ax.scatter(points[[0, -1], 1], points[[0, -1], 0], color=color, s=1)
    # direction arrow on last segment
    if len(points) >= 2:
        ax.annotate(
            '', xy=(points[-1, 1], points[-1, 0]),
            xytext=(points[-2, 1], points[-2, 0]),
            arrowprops=dict(arrowstyle='->', lw=0.6, color=color),
        )


def _draw_topology(ax, lanes, topology):
    """Draw topology connections between centerlines as thin arrows."""
    color = COLOR_DICT['topology']
    topology = np.asarray(topology)
    for i in range(topology.shape[0]):
        for j in range(topology.shape[1]):
            if topology[i, j]:
                pts_i = np.asarray(lanes[i]['points'])
                pts_j = np.asarray(lanes[j]['points'])
                mid_i = pts_i[len(pts_i) // 2]
                mid_j = pts_j[len(pts_j) // 2]
                ax.annotate(
                    '', xy=(mid_j[1], mid_j[0]),
                    xytext=(mid_i[1], mid_i[0]),
                    arrowprops=dict(arrowstyle='->', lw=0.4,
                                    color=color, alpha=0.5),
                )


# ── Public API ────────────────────────────────────────────────────────────

def draw_annotation_bev(annotation, with_topology=False):
    """Render a BEV image of lane centerlines following the LaneSegNet colour
    and layout convention (matplotlib, white background, orange centerlines
    with direction arrows).

    Parameters
    ----------
    annotation : dict
        Must contain ``'lane_centerline'`` (list of dicts each with a
        ``'points'`` key).  Optionally ``'topology_lclc'`` (NxN array).
    with_topology : bool
        If *True* and ``'topology_lclc'`` is present, draw topology arrows.

    Returns
    -------
    np.ndarray  (H, W, 3) uint8 RGB image.
    """
    fig, ax = plt.figure(figsize=(2, 4), dpi=200), plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim([BEV_RANGE[0], BEV_RANGE[1]])
    ax.set_xlim([BEV_RANGE[2], BEV_RANGE[3]])
    ax.invert_xaxis()
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('white')
    fig.tight_layout(pad=0.2)

    lanes = annotation.get('lane_centerline', [])
    for lane in lanes:
        _draw_centerline(ax, lane)

    if with_topology:
        topo = annotation.get('topology_lclc', None)
        if topo is not None and len(lanes) > 0:
            _draw_topology(ax, lanes, topo)

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return data
