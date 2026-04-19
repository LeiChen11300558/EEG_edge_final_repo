#include <Arduino_RouterBridge.h>

String echo_data(String data) {
    return data;
}

void setup() {
    Serial.begin(115200);
    while (!Serial);
    Bridge.begin(460800);
    Bridge.provide("echo_data", echo_data);
    Serial.println("MCU: Echo data ready");
}

void loop() {
    delay(100);
}
