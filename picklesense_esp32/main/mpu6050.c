#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2c_master.h"
#include "driver/gpio.h"

#include "esp_err.h"
#include "esp_check.h"
#include "esp_log.h"

#include "app_config.h"
#include "mpu6050.h"


static const char *TAG = "MPU6050";


// -----------------------------
// MPU6050 Register Addresses
// -----------------------------

#define REG_SMPLRT_DIV      0x19
#define REG_CONFIG          0x1A
#define REG_GYRO_CONFIG     0x1B
#define REG_ACCEL_CONFIG    0x1C
#define REG_ACCEL_XOUT_H    0x3B
#define REG_PWR_MGMT_1      0x6B
#define REG_WHO_AM_I        0x75


// -----------------------------
// Internal Handles
// -----------------------------

static i2c_master_bus_handle_t i2c_bus_handle = NULL;
static i2c_master_dev_handle_t mpu_device_handle = NULL;


// -----------------------------
// Helper: Combine two bytes
// -----------------------------

static int16_t combine_bytes(uint8_t high, uint8_t low)
{
    return (int16_t)(((uint16_t)high << 8) | low);
}


// -----------------------------
// Write one MPU6050 register
// -----------------------------

static esp_err_t mpu6050_write_register(
    uint8_t reg,
    uint8_t value)
{
    uint8_t buffer[2] = {
        reg,
        value
    };

    return i2c_master_transmit(
        mpu_device_handle,
        buffer,
        sizeof(buffer),
        1000
    );
}


// -----------------------------
// Read MPU6050 registers
// -----------------------------

static esp_err_t mpu6050_read_registers(
    uint8_t start_reg,
    uint8_t *data,
    size_t length)
{
    return i2c_master_transmit_receive(
        mpu_device_handle,
        &start_reg,
        1,
        data,
        length,
        1000
    );
}


// -----------------------------
// Initialize I2C Bus
// -----------------------------

static esp_err_t i2c_init(void)
{
    i2c_master_bus_config_t bus_config = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = I2C_SDA_PIN,
        .scl_io_num = I2C_SCL_PIN,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true
    };

    return i2c_new_master_bus(
        &bus_config,
        &i2c_bus_handle
    );
}


// -----------------------------
// Initialize MPU6050
// -----------------------------

esp_err_t mpu6050_init(void)
{
    ESP_LOGI(TAG, "Initializing I2C");

    ESP_RETURN_ON_ERROR(
        i2c_init(),
        TAG,
        "Failed to initialize I2C"
    );


    ESP_LOGI(
        TAG,
        "Checking MPU6050 at address 0x%02X",
        MPU6050_ADDR
    );

    ESP_RETURN_ON_ERROR(
        i2c_master_probe(
            i2c_bus_handle,
            MPU6050_ADDR,
            1000
        ),
        TAG,
        "MPU6050 not detected"
    );


    i2c_device_config_t device_config = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MPU6050_ADDR,
        .scl_speed_hz = I2C_FREQUENCY
    };

    ESP_RETURN_ON_ERROR(
        i2c_master_bus_add_device(
            i2c_bus_handle,
            &device_config,
            &mpu_device_handle
        ),
        TAG,
        "Failed to add MPU6050 device"
    );


    // -----------------------------
    // Verify WHO_AM_I
    // -----------------------------

    uint8_t who_am_i = 0;

    ESP_RETURN_ON_ERROR(
        mpu6050_read_registers(
            REG_WHO_AM_I,
            &who_am_i,
            1
        ),
        TAG,
        "Failed to read WHO_AM_I"
    );

    ESP_LOGI(
        TAG,
        "WHO_AM_I = 0x%02X",
        who_am_i
    );

    if (who_am_i != 0x68)
    {
        ESP_LOGE(
            TAG,
            "Unexpected MPU6050 identity"
        );

        return ESP_FAIL;
    }


    // -----------------------------
    // Reset MPU6050
    // -----------------------------

    ESP_RETURN_ON_ERROR(
        mpu6050_write_register(
            REG_PWR_MGMT_1,
            0x80
        ),
        TAG,
        "Failed to reset MPU6050"
    );

    vTaskDelay(pdMS_TO_TICKS(100));


    // -----------------------------
    // Wake MPU6050
    // Clock source = X-axis gyro
    // -----------------------------

    ESP_RETURN_ON_ERROR(
        mpu6050_write_register(
            REG_PWR_MGMT_1,
            0x01
        ),
        TAG,
        "Failed to wake MPU6050"
    );


    // -----------------------------
    // DLPF configuration
    // CONFIG = 0x03
    // -----------------------------

    ESP_RETURN_ON_ERROR(
        mpu6050_write_register(
            REG_CONFIG,
            0x03
        ),
        TAG,
        "Failed to configure DLPF"
    );


    // -----------------------------
    // Sample Rate
    //
    // 1 kHz / (1 + 9)
    // = 100 Hz
    // -----------------------------

    ESP_RETURN_ON_ERROR(
        mpu6050_write_register(
            REG_SMPLRT_DIV,
            0x09
        ),
        TAG,
        "Failed to configure sample rate"
    );


    // -----------------------------
    // Gyroscope
    //
    // 0x18 = +/-2000 degrees/sec
    // -----------------------------

    ESP_RETURN_ON_ERROR(
        mpu6050_write_register(
            REG_GYRO_CONFIG,
            0x18
        ),
        TAG,
        "Failed to configure gyroscope"
    );


    // -----------------------------
    // Accelerometer
    //
    // 0x00 = +/-2g
    // -----------------------------

    ESP_RETURN_ON_ERROR(
        mpu6050_write_register(
            REG_ACCEL_CONFIG,
            0x00
        ),
        TAG,
        "Failed to configure accelerometer"
    );


    vTaskDelay(pdMS_TO_TICKS(10));


    // -----------------------------
    // Verify ACCEL_CONFIG
    // -----------------------------

    uint8_t accel_config = 0;

    ESP_RETURN_ON_ERROR(
        mpu6050_read_registers(
            REG_ACCEL_CONFIG,
            &accel_config,
            1
        ),
        TAG,
        "Failed to read ACCEL_CONFIG"
    );

    ESP_LOGI(
        TAG,
        "ACCEL_CONFIG = 0x%02X",
        accel_config
    );

    if ((accel_config & 0x18) != 0x00)
    {
        ESP_LOGE(
            TAG,
            "Accelerometer is not configured for +/-2g"
        );

        return ESP_ERR_INVALID_STATE;
    }


    ESP_LOGI(
        TAG,
        "MPU6050 initialized successfully"
    );

    return ESP_OK;
}


// -----------------------------
// Read one six-axis sample
// -----------------------------

esp_err_t mpu6050_read(mpu6050_data_t *data)
{
    if (data == NULL)
    {
        return ESP_ERR_INVALID_ARG;
    }

    uint8_t buffer[14];

    ESP_RETURN_ON_ERROR(
        mpu6050_read_registers(
            REG_ACCEL_XOUT_H,
            buffer,
            sizeof(buffer)
        ),
        TAG,
        "Failed to read motion data"
    );


    // Accelerometer
    data->ax = combine_bytes(
        buffer[0],
        buffer[1]
    );

    data->ay = combine_bytes(
        buffer[2],
        buffer[3]
    );

    data->az = combine_bytes(
        buffer[4],
        buffer[5]
    );


    // buffer[6] and buffer[7]
    // are temperature data


    // Gyroscope
    data->gx = combine_bytes(
        buffer[8],
        buffer[9]
    );

    data->gy = combine_bytes(
        buffer[10],
        buffer[11]
    );

    data->gz = combine_bytes(
        buffer[12],
        buffer[13]
    );


    return ESP_OK;
}