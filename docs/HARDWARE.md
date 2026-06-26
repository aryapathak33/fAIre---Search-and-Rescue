# fAIre hardware notes

The hardware side is intentionally simple so the software can be tested without requiring expensive robotics parts. The current sketch streams sensor readings over serial and lets the Python side decide how to use them.

## Current hardware concept

- Small rover / robot chassis
- Camera or webcam for visual input
- Microcontroller for sensor polling
- Temperature sensor
- Smoke / gas sensor
- Optional CO sensor
- Optional VOC / combustible gas sensor
- Optional oxygen or CO2 sensor for better chemistry awareness
- Ultrasonic distance sensor for obstacle proximity

## Serial format

The Arduino sketch prints one comma-separated reading per line:

```text
temperature_c,smoke_raw,co_raw,voc_raw,distance_cm
```

Example:

```text
42.31,510,390,275,84.20
```

These raw values can be parsed by Python and passed into the risk engine.

## Why these sensors?

- **Camera** gives the AI model visual context.
- **Temperature** helps identify dangerous heat regions.
- **Smoke / gas** gives a basic proxy for poor visibility and combustion products.
- **CO** is useful because incomplete combustion creates toxic carbon monoxide.
- **VOC / combustible gas** can help study pyrolysis gases from heated materials.
- **Oxygen / CO2** would make the flashover module more meaningful, but these sensors need calibration.
- **Distance** helps the rover avoid obstacles.

## Calibration warning

MQ-style analog gas sensors are good for learning and demos, but their raw numbers should not be treated as exact ppm without calibration. For a serious fire-safety system, gas sensors would need calibration curves, temperature/humidity compensation, known sensor placement, and validation against real fire-test data.

## Future hardware upgrades

- Thermal camera
- Calibrated CO sensor
- Calibrated CO2 sensor
- Oxygen sensor
- Heat-flux sensor
- Better chassis and motor driver
- Live serial parsing inside `inference/detect.py`
- Onboard compute using Raspberry Pi or Jetson-style hardware
