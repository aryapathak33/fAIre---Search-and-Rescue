/*
  fAIre sensor streaming sketch

  Streams simple sensor readings over Serial so the Python inference pipeline can
  combine visual model confidence with hardware risk signals.

  Output format:
    temperature_c,smoke_raw,distance_cm

  Notes:
  - This sketch uses no external libraries so it is easy to upload.
  - Replace pins/sensor conversion formulas with your exact hardware.
*/

const int TRIG_PIN = 9;
const int ECHO_PIN = 10;
const int SMOKE_PIN = A0;
const int TEMP_PIN = A1;  // Example: LM35-style analog temperature sensor

const unsigned long SAMPLE_DELAY_MS = 250;

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

float readDistanceCm() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000); // timeout after 30 ms
  if (duration == 0) {
    return -1.0; // no echo detected
  }

  // Speed of sound approximation: 0.0343 cm/us, divide by 2 for round trip.
  return duration * 0.0343 / 2.0;
}

float readTemperatureC() {
  int raw = analogRead(TEMP_PIN);
  float voltage = raw * (5.0 / 1023.0);

  // LM35 example conversion: 10 mV per degree C.
  // Replace this if your sensor is different.
  return voltage * 100.0;
}

void loop() {
  float temperatureC = readTemperatureC();
  int smokeRaw = analogRead(SMOKE_PIN);
  float distanceCm = readDistanceCm();

  Serial.print(temperatureC, 2);
  Serial.print(",");
  Serial.print(smokeRaw);
  Serial.print(",");
  Serial.println(distanceCm, 2);

  delay(SAMPLE_DELAY_MS);
}
