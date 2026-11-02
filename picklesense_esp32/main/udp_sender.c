#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <inttypes.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#include "esp_log.h"

#include "app_config.h"
#include "udp_sender.h"


static const char *TAG = "UDP_SENDER";

static int udp_socket = -1;
static struct sockaddr_in destination_addr;


/*
 * Retry configuration.
 *
 * A delay of 1 FreeRTOS tick is used between retries.
 *
 * This gives the Wi-Fi driver time to release TX buffers
 * instead of immediately hammering sendto() again.
 */
#define UDP_SEND_MAX_RETRIES 10


esp_err_t udp_sender_init(void)
{
    udp_socket =
        socket(
            AF_INET,
            SOCK_DGRAM,
            IPPROTO_UDP
        );

    if (udp_socket < 0)
    {
        ESP_LOGE(
            TAG,
            "Failed to create UDP socket: errno=%d (%s)",
            errno,
            strerror(errno)
        );

        return ESP_FAIL;
    }


    memset(
        &destination_addr,
        0,
        sizeof(destination_addr)
    );


    destination_addr.sin_family =
        AF_INET;

    destination_addr.sin_port =
        htons(UDP_PORT);


    int result =
        inet_pton(
            AF_INET,
            LAPTOP_IP,
            &destination_addr.sin_addr
        );


    if (result != 1)
    {
        ESP_LOGE(
            TAG,
            "Invalid laptop IP address: %s",
            LAPTOP_IP
        );

        close(udp_socket);

        udp_socket = -1;

        return ESP_ERR_INVALID_ARG;
    }


    ESP_LOGI(
        TAG,
        "UDP sender initialized"
    );

    ESP_LOGI(
        TAG,
        "Destination: %s:%d",
        LAPTOP_IP,
        UDP_PORT
    );


    return ESP_OK;
}


esp_err_t udp_sender_send(
    uint32_t sequence,
    int64_t timestamp_us,

    int16_t ax,
    int16_t ay,
    int16_t az,

    int16_t gx,
    int16_t gy,
    int16_t gz
)
{
    if (udp_socket < 0)
    {
        return ESP_ERR_INVALID_STATE;
    }


    char packet[160];


    int length =
        snprintf(
            packet,
            sizeof(packet),

            "%" PRIu32 ","
            "%" PRId64 ","
            "%d,%d,%d,"
            "%d,%d,%d",

            sequence,
            timestamp_us,

            ax,
            ay,
            az,

            gx,
            gy,
            gz
        );


    if (
        length < 0 ||
        length >= sizeof(packet)
    )
    {
        ESP_LOGE(
            TAG,
            "UDP packet formatting failed"
        );

        return ESP_FAIL;
    }


    /*
     * sendto() may temporarily return ENOMEM when
     * the Wi-Fi driver's TX buffers are full.
     *
     * Instead of immediately dropping the IMU sample,
     * wait briefly and retry.
     */
    for (
        int attempt = 0;
        attempt <= UDP_SEND_MAX_RETRIES;
        attempt++
    )
    {
        int sent =
            sendto(
                udp_socket,
                packet,
                length,
                0,

                (struct sockaddr *)&destination_addr,
                sizeof(destination_addr)
            );


        /*
         * Entire UDP packet was accepted.
         */
        if (sent == length)
        {
            return ESP_OK;
        }


        /*
         * TX buffer temporarily unavailable.
         */
        if (
            sent < 0 &&
            errno == ENOMEM
        )
        {
            if (
                attempt <
                UDP_SEND_MAX_RETRIES
            )
            {
                /*
                 * Allow Wi-Fi driver TX buffers
                 * to drain before retrying.
                 */
                vTaskDelay(1);

                continue;
            }


            /*
             * Only log after all retries failed.
             *
             * This prevents hundreds of warning
             * messages from flooding the serial port.
             */
            ESP_LOGW(
                TAG,
                "UDP send failed after %d retries: "
                "errno=%d (%s)",
                UDP_SEND_MAX_RETRIES,
                errno,
                strerror(errno)
            );

            return ESP_FAIL;
        }


        /*
         * Any other socket error is unexpected.
         */
        if (sent < 0)
        {
            ESP_LOGW(
                TAG,
                "UDP send failed: errno=%d (%s)",
                errno,
                strerror(errno)
            );

            return ESP_FAIL;
        }


        /*
         * UDP should normally send the entire datagram.
         */
        ESP_LOGW(
            TAG,
            "UDP partial send: %d of %d bytes",
            sent,
            length
        );

        return ESP_FAIL;
    }


    return ESP_FAIL;
}