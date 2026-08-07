/*
 * ESP32-S3 GPT Inference Demo (Macbeth GPT)
 * 
 * Hardware Requirements:
 * - ESP32-S3-WROOM-1 Module (with PSRAM enabled in Arduino IDE)
 * - 1.3" OLED I2C Display (SH1106 or SSD1306)
 * 
 * Arduino IDE Settings:
 * - Board: ESP32S3 Dev Module
 * - PSRAM: OPI PSRAM (or QPI depending on your board, check if it has PSRAM)
 * - Partition Scheme: Default 4MB with spiffs or 8MB/16MB depending on your board flash
 *
 * Make sure to install the "U8g2" library via the Arduino Library Manager!
 */

#include <Arduino.h>
#include <Wire.h>
#include <U8x8lib.h>
#include <FS.h>
#include <LittleFS.h>
#include "GPTInference.hpp"

// --- Hardware Pins Configuration ---
#define OLED_SDA 17
#define OLED_SCL 18

// --- Display Constructor Choice ---
// For 1.3" OLEDs, SH1106 is the most common driver.
U8X8_SH1106_128X64_NONAME_HW_I2C u8x8(/* reset=*/ U8X8_PIN_NONE, /* clock=*/ OLED_SCL, /* data=*/ OLED_SDA);

// Setup U8X8LOG for automatic wrapping and scrolling
U8X8LOG u8x8log;
#define U8LOG_WIDTH 16
#define U8LOG_HEIGHT 8
uint8_t u8log_buffer[U8LOG_WIDTH * U8LOG_HEIGHT];

GPTInference model;
int* tokens = nullptr;
int tokens_count = 0;
float* logits = nullptr;

// Helper function to print to Serial and OLED
void display_char(char c) {
  Serial.print(c);
  if (c == '\r') return;
  u8x8log.print(c);
}

void setup() {
  Serial.begin(115200);
  // Wait up to 3 seconds for Serial connection to be established (only on native USB)
  for (int i = 0; i < 30; ++i) {
    if (Serial) break;
    delay(100);
  }
  
  Serial.println("\n==================================");
  Serial.println("ESP32-S3 Macbeth GPT Inference Engine");
  Serial.println("==================================");

  // Initialize display
  u8x8.begin();
  u8x8.setFont(u8x8_font_chroma48medium8_r);
  u8x8.drawString(0, 0, "ESP32 GPT Init...");
  u8x8.drawString(0, 2, "Mounting FS...");

  // Mount LittleFS filesystem
  if (!LittleFS.begin(true)) {
    Serial.println("[-] Error: LittleFS mount failed!");
    u8x8.drawString(0, 3, "FS Mount ERROR!");
    while (1) delay(1000);
  }
  Serial.println("[+] LittleFS mounted successfully.");
  u8x8.drawString(0, 2, "FS Mounted.    ");
  u8x8.drawString(0, 4, "Loading model...");

  // Check if model file exists
  if (!LittleFS.exists("/macbeth.bin")) {
    Serial.println("[-] Error: /macbeth.bin not found in LittleFS!");
    Serial.println("[!] Please upload the model file using PlatformIO.");
    u8x8.drawString(0, 4, "macbeth.bin not");
    u8x8.drawString(0, 5, "found in LittleFS");
    while (1) delay(1000);
  }

  // Load GPT Model from LittleFS
  if (!model.load_model("/littlefs/macbeth.bin")) {
    Serial.println("[-] Error: Failed to load GPT model weights!");
    u8x8.drawString(0, 4, "Load weight ERR");
    while (1) delay(1000);
  }

  u8x8.clear();
  u8x8.drawString(0, 0, "Model loaded!");
  u8x8.drawString(0, 2, "Vocab Size: ");
  u8x8.print(model.config.vocab_size);
  u8x8.drawString(0, 3, "Layers:     ");
  u8x8.print(model.config.num_layers);
  u8x8.drawString(0, 4, "Dim:        ");
  u8x8.print(model.config.embedding_dim);
  
  delay(3000);
  u8x8.clear();

  // Initialize U8X8LOG text console
  u8x8log.begin(u8x8, U8LOG_WIDTH, U8LOG_HEIGHT, u8log_buffer);
  u8x8log.setRedrawMode(0); // 0 = redraw line-by-line, fast!

  // Allocate token buffer and logits buffer
  tokens = (int*)malloc(model.config.max_seq_len * sizeof(int));
  logits = (float*)malloc(model.config.vocab_size * sizeof(float));
  if (!tokens || !logits) {
    Serial.println("[-] Error: Failed to allocate inference arrays.");
    while (1);
  }

  // Define starting prompt
  const char* prompt = "The ";
  Serial.print("[*] Prompt: ");
  Serial.println(prompt);

  // Tokenize prompt
  tokens_count = 0;
  for (int i = 0; prompt[i] != '\0'; ++i) {
    char c = prompt[i];
    int token_id = -1;
    for (int j = 0; j < model.config.vocab_size; ++j) {
      if (model.id_to_char[j] == c) {
        token_id = j;
        break;
      }
    }
    if (token_id == -1) token_id = 0; // Default fallback
    tokens[tokens_count++] = token_id;
    display_char(c);
  }

  randomSeed(analogRead(0));
}

void loop() {
  // 1. Run forward pass to get logits for the last token
  model.forward(tokens, tokens_count, logits);

  // 2. Softmax & Temperature Sampling
  float temperature = 0.9f;
  float max_logit = logits[0];
  for (int j = 1; j < model.config.vocab_size; ++j) {
    if (logits[j] > max_logit) max_logit = logits[j];
  }

  float sum_exp = 0.0f;
  for (int j = 0; j < model.config.vocab_size; ++j) {
    // Apply temperature scaling
    logits[j] = (logits[j] - max_logit) / temperature;
    logits[j] = exp(logits[j]);
    sum_exp += logits[j];
  }

  // Draw random sample
  float r = ((float)random(0, 100000) / 100000.0f) * sum_exp;
  float running_sum = 0.0f;
  int next_token = 0;
  for (int j = 0; j < model.config.vocab_size; ++j) {
    running_sum += logits[j];
    if (r <= running_sum) {
      next_token = j;
      break;
    }
  }

  // 3. Print the token's character representation
  char next_char = model.id_to_char[next_token];
  display_char(next_char);

  // 4. Update prompt token window
  if (tokens_count >= model.config.max_seq_len) {
    // Shift window left
    for (int i = 0; i < model.config.max_seq_len - 1; ++i) {
      tokens[i] = tokens[i + 1];
    }
    tokens[model.config.max_seq_len - 1] = next_token;
  } else {
    tokens[tokens_count++] = next_token;
  }

  // Slow down slightly for readability on screen
  delay(10);
}
