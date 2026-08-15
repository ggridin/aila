# AILA

AILA is a host-independent Python core for an isolated Hermes Agent wake loop
with a small v1 body. The brain-facing body is fixed to five workers: `mic`,
`camera`, `filesystem`, `speaker`, and `display`. Physical microphone and camera
ownership sits below that boundary in `audio-input` and `camera-input` device
services; the device services are consumed by workers and are not brain-facing.
