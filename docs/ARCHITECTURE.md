# Architecture

End-to-end flow:

```
[ sensors / camera ]
        |
        v
[ preprocessing ]  (data/)
        |
        v
[ detection model ]  (inference/, weights in models/)
        |
        v
[ alerting + real-time relay to firefighters ]
```

## Components
- **Sensing** — `[FILL IN: brief]` (see HARDWARE.md)
- **Perception model** — `[FILL IN: architecture]` (see TRAINING.md)
- **Flashover-risk logic** — `[FILL IN: how the imminent-flashover alert is triggered]`
- **Output / relay** — `[FILL IN: how data gets back to firefighters in real time]`

## Design decisions
`[FILL IN: 2–3 choices you made and why — interviewers love this section.
e.g. why this model size, why this sensor, latency vs accuracy tradeoffs.]`
