#ifndef APP_CONFIG_H
#define APP_CONFIG_H

// -----------------------------
// Wi-Fi Configuration
// -----------------------------

#define WIFI_SSID       "PisoWifi"
#define WIFI_PASSWORD   "Mahalbayad123!"

// -----------------------------
// Laptop UDP Destination
// -----------------------------

#define LAPTOP_IP       "192.168.1.10"
#define UDP_PORT        5005

// -----------------------------
// MPU6050 / I2C Configuration
// -----------------------------

#define I2C_SDA_PIN     21
#define I2C_SCL_PIN     22
#define I2C_FREQUENCY   100000

#define MPU6050_ADDR    0x68

// -----------------------------
// Sampling Configuration
// -----------------------------

#define SAMPLE_RATE_HZ      100
#define SAMPLE_PERIOD_MS    10

#endif