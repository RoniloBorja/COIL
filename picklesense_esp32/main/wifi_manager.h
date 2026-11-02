#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <stdbool.h>
#include "esp_err.h"

// Initialize Wi-Fi in station mode and start connecting
esp_err_t wifi_manager_init(void);

// Returns true once the ESP32 has obtained an IP address
bool wifi_manager_is_connected(void);

#endif