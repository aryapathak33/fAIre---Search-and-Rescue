# Hardware build

The hardware side of fAIre is the robot/sensor layer that feeds the AI pipeline.

## Prototype concept

The system is designed around a small rover or portable robot platform that can move into an unsafe area before a firefighter enters. The robot streams camera frames and sensor readings to the perception pipeline.

## Suggested component table

Replace the details below with your exact parts once you confirm them.

| Component | Role in the system | Notes |
|---|---|---|
| Arduino / microcontroller | Reads low-level sensors | Streams values over serial |
| Camera | Provides frames for the CNN | USB camera, Pi camera, or phone/camera feed |
| Ultrasonic distance sensor | Estimates distance to obstacle/wall | Useful for rover navigation prototype |
| Smoke / gas sensor | Adds environmental risk signal | Analog value can be normalized in software |
| Temperature sensor | Adds heat severity signal | Optional, but useful for risk scoring |
| Rover chassis / motors | Physical mobility | Can be documented even before full autonomy |

## Serial sensor stream

The included Arduino sketch prints comma-separated sensor readings:

```text
temperature_c,smoke_raw,distance_cm
```

Example:

```text
31.2,420,85.4
```

The inference code can later parse that stream and pass the values into `inference/risk_engine.py`.

## 3D / spatial sensing story

If your original prototype used sensor sweeps to build a rough 3D map, explain it like this:

> The early prototype sampled distance readings at different robot/sensor angles and used those readings to reconstruct a rough spatial profile of the environment. Instead of relying only on a camera frame, the robot could estimate where walls or obstacles were relative to the sensor. That early experiment motivated the current system design: combine visual AI with physical sensor data instead of treating the model as the whole robot.

Add your exact method once you verify it:

- what sensor you used,
- how it moved or scanned,
- what each reading represented,
- and how you visualized/reconstructed space.

## Media to add

Add these to `media/`:

- photo of the robot,
- wiring photo,
- screenshot of serial output,
- demo GIF or short video.
