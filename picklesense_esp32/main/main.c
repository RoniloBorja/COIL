#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "app_config.h"
#include "mpu6050.h"
#include "wifi_manager.h"
#include "udp_sender.h"

static const char *TAG = "PICKLESENSE";

typedef struct
{
    uint32_t sequence;
    int64_t timestamp_us;

    int16_t ax;
    int16_t ay;
    int16_t az;

    int16_t gx;
    int16_t gy;
    int16_t gz;

} imu_packet_t;

static QueueHandle_t imu_queue = NULL;

static uint32_t dropped_queue_samples = 0;
static uint32_t udp_send_failures = 0;


/*
 * MPU6050 acquisition task
 *
 * Runs independently from networking.
 * Samples sensor at the configured rate and places
 * the result into a FreeRTOS queue.
 */
static void imu_acquisition_task(void *arg)
{
    uint32_t sequence = 0;

    const TickType_t sample_period =
        pdMS_TO_TICKS(SAMPLE_PERIOD_MS);

    TickType_t last_wake_time =
        xTaskGetTickCount();

    ESP_LOGI(
        TAG,
        "IMU acquisition task started at %d Hz",
        SAMPLE_RATE_HZ
    );

    while (1)
    {
        mpu6050_data_t sensor_data;

        esp_err_t result =
            mpu6050_read(&sensor_data);

        if (result == ESP_OK)
        {
            imu_packet_t packet;

            packet.sequence = sequence;
            packet.timestamp_us = esp_timer_get_time();

            packet.ax = sensor_data.ax;
            packet.ay = sensor_data.ay;
            packet.az = sensor_data.az;

            packet.gx = sensor_data.gx;
            packet.gy = sensor_data.gy;
            packet.gz = sensor_data.gz;

            /*
             * Do not block the acquisition task.
             *
             * If networking falls behind and the queue
             * becomes full, count the dropped sample.
             */
            if (
                xQueueSend(
                    imu_queue,
                    &packet,
                    0
                ) != pdPASS
            )
            {
                dropped_queue_samples++;
            }

            sequence++;
        }
        else
        {
            ESP_LOGW(
                TAG,
                "MPU6050 read failed: %s",
                esp_err_to_name(result)
            );
        }

        /*
         * Fixed-rate scheduling.
         */
        vTaskDelayUntil(
            &last_wake_time,
            sample_period
        );
    }
}


/*
 * UDP sender task
 *
 * Waits for samples from the acquisition queue
 * and sends them to the laptop.
 */
static void udp_sender_task(void *arg)
{
    imu_packet_t packet;

    ESP_LOGI(
        TAG,
        "UDP sender task started"
    );

    while (1)
    {
        if (
            xQueueReceive(
                imu_queue,
                &packet,
                portMAX_DELAY
            ) == pdPASS
        )
        {
            if (wifi_manager_is_connected())
            {
                esp_err_t result =
                    udp_sender_send(
                        packet.sequence,
                        packet.timestamp_us,

                        packet.ax,
                        packet.ay,
                        packet.az,

                        packet.gx,
                        packet.gy,
                        packet.gz
                    );

                if (result != ESP_OK)
                {
                    udp_send_failures++;
                }
            }
        }
    }
}


/*
 * Statistics task
 *
 * Prints periodic status instead of printing
 * every individual sensor sample.
 */
static void statistics_task(void *arg)
{
    while (1)
    {
        vTaskDelay(
            pdMS_TO_TICKS(5000)
        );

        UBaseType_t queue_used =
            uxQueueMessagesWaiting(imu_queue);

        ESP_LOGI(
            TAG,
            "STATUS | queue=%u | queue_drops=%" PRIu32
            " | udp_failures=%" PRIu32,
            (unsigned int)queue_used,
            dropped_queue_samples,
            udp_send_failures
        );
    }
}


void app_main(void)
{
    ESP_LOGI(
        TAG,
        "Starting PickleSense"
    );

    ESP_LOGI(
        TAG,
        "Initializing MPU6050..."
    );

    ESP_ERROR_CHECK(
        mpu6050_init()
    );

    ESP_LOGI(
        TAG,
        "Connecting to Wi-Fi..."
    );

    ESP_ERROR_CHECK(
        wifi_manager_init()
    );

    ESP_LOGI(
        TAG,
        "Initializing UDP sender..."
    );

    ESP_ERROR_CHECK(
        udp_sender_init()
    );


    /*
     * Queue capacity:
     *
     * 50 samples = 0.5 seconds of buffering
     * at 100 Hz.
     */
    imu_queue =
        xQueueCreate(
            50,
            sizeof(imu_packet_t)
        );

    if (imu_queue == NULL)
    {
        ESP_LOGE(
            TAG,
            "Failed to create IMU queue"
        );

        return;
    }


    /*
     * Acquisition task
     *
     * Higher priority than UDP sender because
     * maintaining stable sensor timing is critical.
     */
    xTaskCreate(
        imu_acquisition_task,
        "imu_acquisition",
        4096,
        NULL,
        6,
        NULL
    );


    /*
     * Networking task
     */
    xTaskCreate(
        udp_sender_task,
        "udp_sender",
        4096,
        NULL,
        5,
        NULL
    );


    /*
     * Low-priority statistics task
     */
    xTaskCreate(
        statistics_task,
        "statistics",
        3072,
        NULL,
        2,
        NULL
    );


    ESP_LOGI(
        TAG,
        "PickleSense tasks started"
    );
}