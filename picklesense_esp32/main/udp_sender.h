#ifndef UDP_SENDER_H
#define UDP_SENDER_H

#include <stdint.h>
#include "esp_err.h"

// Initialize the UDP sender
esp_err_t udp_sender_init(void);

// Send one MPU6050 sample to the laptop
esp_err_t udp_sender_send(
    uint32_t sequence,
    int64_t timestamp_us,
    int16_t ax,
    int16_t ay,
    int16_t az,
    int16_t gx,
    int16_t gy,
    int16_t gz
);

#endif