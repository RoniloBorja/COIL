#ifndef MPU6050_H
#define MPU6050_H

#include <stdint.h>
#include "esp_err.h"

// Structure for one MPU6050 sample
typedef struct
{
    int16_t ax;
    int16_t ay;
    int16_t az;

    int16_t gx;
    int16_t gy;
    int16_t gz;

} mpu6050_data_t;


// Initialize I2C and MPU6050
esp_err_t mpu6050_init(void);

// Read one 6-axis sample
esp_err_t mpu6050_read(mpu6050_data_t *data);

#endif