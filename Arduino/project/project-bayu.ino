#include <WiFi.h>
#include <WebServer.h>

WebServer server(80);

const char* ssid = "sehat";
const char* password = "yokbisayok";

const int vibsPin = 12;
const int buzzPin = 13;
const int ledPin = 15;

void setup() {
    Serial.begin(115200);

    // Wi-Fi connection
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }  
    Serial.println("");
    Serial.println("WiFi connected");
    
    Serial.println("ESP32-CAM IP Address: " + WiFi.localIP().toString());

    server.on("/control_sensors", HTTP_GET, [](){
        String action = server.arg("action");
        controlSensors(action);
        server.send(200, "text/plain", "Sensors controlled");
    });

    server.begin();
}

void loop() {
    server.handleClient();
}

void controlSensors(String action) {
    if (action.equals("activate")) {
        activateSensors();
    } else if (action.equals("deactivate")) {
        deactivateSensors();
    }
}

void activateSensors() {
    Serial.println("Sensor aktif");
    digitalWrite(buzzPin, HIGH);
    for (int i = 0; i < 5; i++) {
        digitalWrite(ledPin, HIGH);
        delay(500);
        digitalWrite(ledPin, LOW);
        delay(500);
    }
    digitalWrite(vibsPin, HIGH);
}

void deactivateSensors() {
    Serial.println("Sensor tidak aktif!");
    digitalWrite(buzzPin, LOW);
    digitalWrite(ledPin, LOW);
    digitalWrite(vibsPin, LOW);
}
