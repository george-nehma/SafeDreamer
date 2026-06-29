import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R


# 1. Load data
file_path = "/workspace/SafeDreamer/logs/lander_6dof_direct/20260217_164140/train_eps/20260217T181058-42001926b6db43169d8fdf5804a09d13-346.npz"
data = np.load(file_path)
state = data['state']

# Extract Quaternions (0:4) and Position (4:7)
# IsaacLab usually uses [w, x, y, z]
quats = state[:, 0:4] 
pos = state[:, 4:7]

fig = go.Figure()

# 2. Plot the Trajectory Line
fig.add_trace(go.Scatter3d(
    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
    mode='lines',
    line=dict(color='cyan', width=3),
    name='Flight Path'
))

# 3. Add Start and End Markers
fig.add_trace(go.Scatter3d(
    x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
    mode='markers',
    marker=dict(size=8, color='green', symbol='diamond'),
    name='Start'
))

fig.add_trace(go.Scatter3d(
    x=[pos[-1, 0]], y=[pos[-1, 1]], z=[pos[-1, 2]],
    mode='markers',
    marker=dict(size=8, color='red', symbol='square'),
    name='End'
))

# 4. Add Orientation Vectors (every 10th frame)
step = 10 
vec_len = 0.4 
colours = ['red', 'green', 'blue']
for i in range(0, len(pos), step):
    try:
        # Assuming [w, x, y, z] order
        r = R.from_quat(quats[i], scalar_first=True) 
        # Apply rotation to a forward-pointing vector [vec_len, 0, 0]
        for j in range(0,4):
            dir_vec = [0, 0, 0]
            dir_vec[j] = vec_len
            direction = r.apply(dir_vec)
            
            fig.add_trace(go.Scatter3d(
                x=[pos[i, 0], pos[i, 0] + direction[0]],
                y=[pos[i, 1], pos[i, 1] + direction[1]],
                z=[pos[i, 2], pos[i, 2] + direction[2]],
                mode='lines',
                line=dict(color=colours[j], width=2),
                showlegend=False,
                hoverinfo='none'
        ))
    except Exception:
        continue

# 5. Layout and Styling
fig.update_layout(
    title="Lander 6DOF: Final Trajectory Visualisation",
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='data'
    ),
    template="plotly_dark",
    legend=dict(yanchor="top", y=0.9, xanchor="left", x=0.1)
)

fig.show()