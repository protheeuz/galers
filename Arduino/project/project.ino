#include <Wire.h>
#include <WiFi.h>
//#include <ESPAsyncWebServer.h>
#include <ESPAsyncWebSrv.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include "esp_camera.h"

int ledPin = 13;
int buzzerPin = 9;
int vibrationMotorPin = 6;

const char *ssid = "MathtechStudio";
const char *password = "Bellaciao13";

AsyncWebServer server(80);

const char *serverFlask = "http://127.0.0.1:5000/capture_frame";

void captureAndSendFrame() {
    // Ambil gambar dari kamera
  camera_fb_t *fb = esp_camera_fb_get();
  
  if (fb) {
    // Ubah gambar menjadi JSON
    StaticJsonDocument<20000> doc;
    JsonArray frameArray = doc.to<JsonArray>();
    
    for (size_t i = 0; i < fb->len; i++) {
      frameArray.add(fb->buf[i]);
    }

    // Kirim gambar ke server Flask
    HTTPClient http;
    http.begin(serverAddress);
    http.addHeader("Content-Type", "application/json");
    int httpResponseCode = http.POST(doc.as<String>());
    
    if (httpResponseCode > 0) {
      Serial.print("Kode respons server: ");
      Serial.println(httpResponseCode);
      
      // Tanggapi jika diperlukan
      String response = http.getString();
      Serial.println(response);
    } else {
      Serial.print("Gagal mengirimkan frame. Kode respons: ");
      Serial.println(httpResponseCode);
    }

    // Hapus buffer gambar setelah selesai digunakan
    esp_camera_fb_return(fb);
  } else {
    Serial.println("Gagal mengambil frame dari kamera");
  }
}

//LiquidCrystal_I2C lcd(0x27, 16, 2);

void setup() {
  Serial.begin(115200);

  // Inisialisasi kamera
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    
    delay(500);
    Serial.println("Koneksi ke WiFi...");
  }
  Serial.println("WiFi terkoneksi");

  esp_err_t res = esp_camera_init(&config);
  if (res != ESP_OK) {
    Serial.printf("Gagal menginisialisasi kamera. Kode kesalahan: %d", res);
    return;
  }

  while (true) {
    captureAndSendFrame();
    delay(5000);
  }

  Serial.print("IP Address: ");
  Serial.println(WiFi.localIP());


  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
  String html = "<html><body><h1>ESP32 IP Address</h1><p>IP Address: " + WiFi.localIP().toString() + "</p></body></html>";
  request->send(200, "text/html", html);
  });

  pinMode(ledPin, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  //  pinMode(relayPin, OUTPUT);
  pinMode(vibrationMotorPin, OUTPUT);

  //  lcd.begin(16, 2);
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
//  digitalWrite(relayPin, HIGH);
  digitalWrite(vibrationMotorPin, HIGH);

//  lcd.clear();
//  lcd.print("Terdeteksi kantuk!!!");
  delay(2000);

  // Matikan alarm setelah beberapa detik (sesuaikan dengan kebutuhan)
  deactivateAlarm();
}

void deactivateAlarm() {
  digitalWrite(ledPin, LOW);
  digitalWrite(buzzerPin, LOW);
//  digitalWrite(relayPin, LOW);
  digitalWrite(vibrationMotorPin, LOW);

//  lcd.clear();
//  lcd.print("Tidak terdeteksi kantuk");
}
