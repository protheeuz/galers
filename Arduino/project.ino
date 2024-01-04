#include <Wire.h>
#include <WiFi.h>
#include <ESPAsyncWebServer.h>

int ledPin = 13;
int buzzerPin = 9;
int vibrationMotorPin = 6;

const char *ssid = "NAMA_WIFI";
const char *password = "PASSWORD_WIFI";

AsyncWebServer server(80);

LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Koneksi ke WiFi...");
  }
  Serial.println("WiFi terkoneksi");

  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());


  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
  String html = "<html><body><h1>ESP32 IP Address</h1><p>IP Address: " + WiFi.localIP().toString() + "</p></body></html>";
  request->send(200, "text/html", html);
  });

  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  pinMode(relayPin, OUTPUT);
  pinMode(vibrationMotorPin, OUTPUT);

  lcd.begin(16, 2);
  server.begin();
}

void loop() {
  if (Serial.available() > 0) {
    char data = Serial.read();

    // Respons tergantung pada sinyal yang diterima
    if (data == '1') {
      activateAlarm();
    } else {
      deactivateAlarm();
    }
  }
}

void activateAlarm() {
  digitalWrite(ledPin, HIGH);
  digitalWrite(buzzerPin, HIGH);
  digitalWrite(relayPin, HIGH);
  digitalWrite(vibrationMotorPin, HIGH);

  lcd.clear();
  lcd.print("Terdeteksi kantuk!!!");
  delay(2000);

  // Matikan alarm setelah beberapa detik (sesuaikan dengan kebutuhan)
  deactivateAlarm();
}

void deactivateAlarm() {
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin, LOW);
  digitalWrite(relayPin, LOW);
  digitalWrite(vibrationMotorPin, LOW);

  lcd.clear();
  lcd.print("Tidak terdeteksi kantuk");
}
