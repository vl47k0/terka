"""terka — THETIS-to-rakija pose-trajectory converter.

Walks the THETIS dataset (1980 RGB videos of 12 tennis-shot
classes, 55 subjects), runs MediaPipe Pose Landmarker on each
frame, maps the 33 landmarks onto rakija's 18-joint PoseJoints
schema, and POSTs the resulting trajectory JSON to vertex's
/trajectories/ endpoint.

The output JSON matches rakija's pose_rig_trajectory_load_json
exactly — same shape kadar produces — so rakija's existing 'Load
Latest from Vertex' right-click flow picks it up unchanged.
"""
