#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "esp_check.h"
#include "nvs_flash.h"

#include "app_config.h"
#include "wifi_manager.h"


static const char *TAG = "WIFI_MANAGER";

static EventGroupHandle_t wifi_event_group;

#define WIFI_CONNECTED_BIT BIT0

static bool connected = false;


// ---------------------------------
// Wi-Fi / IP event handler
// ---------------------------------

static void wifi_event_handler(
    void *arg,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data)
{
    if (event_base == WIFI_EVENT &&
        event_id == WIFI_EVENT_STA_START)
    {
        ESP_LOGI(TAG, "Wi-Fi station started");

        esp_wifi_connect();
    }

    else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED)
    {
        connected = false;

        xEventGroupClearBits(
            wifi_event_group,
            WIFI_CONNECTED_BIT
        );

        ESP_LOGW(
            TAG,
            "Wi-Fi disconnected, reconnecting..."
        );

        esp_wifi_connect();
    }

    else if (event_base == IP_EVENT &&
             event_id == IP_EVENT_STA_GOT_IP)
    {
        ip_event_got_ip_t *event =
            (ip_event_got_ip_t *)event_data;

        connected = true;

        xEventGroupSetBits(
            wifi_event_group,
            WIFI_CONNECTED_BIT
        );

        ESP_LOGI(
            TAG,
            "Wi-Fi connected"
        );

        ESP_LOGI(
            TAG,
            "ESP32 IP address: " IPSTR,
            IP2STR(&event->ip_info.ip)
        );
    }
}


// ---------------------------------
// Initialize Wi-Fi
// ---------------------------------

esp_err_t wifi_manager_init(void)
{
    esp_err_t ret;


    // ---------------------------------
    // Initialize NVS
    // Required by Wi-Fi
    // ---------------------------------

    ret = nvs_flash_init();

    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND)
    {
        ESP_ERROR_CHECK(
            nvs_flash_erase()
        );

        ret = nvs_flash_init();
    }

    ESP_RETURN_ON_ERROR(
        ret,
        TAG,
        "Failed to initialize NVS"
    );


    // ---------------------------------
    // Initialize TCP/IP stack
    // ---------------------------------

    ESP_RETURN_ON_ERROR(
        esp_netif_init(),
        TAG,
        "Failed to initialize esp_netif"
    );


    // ---------------------------------
    // Create default event loop
    // ---------------------------------

    ESP_RETURN_ON_ERROR(
        esp_event_loop_create_default(),
        TAG,
        "Failed to create event loop"
    );


    // ---------------------------------
    // Create default Wi-Fi station
    // ---------------------------------

    esp_netif_create_default_wifi_sta();


    // ---------------------------------
    // Initialize Wi-Fi driver
    // ---------------------------------

    wifi_init_config_t wifi_init_config =
        WIFI_INIT_CONFIG_DEFAULT();

    ESP_RETURN_ON_ERROR(
        esp_wifi_init(&wifi_init_config),
        TAG,
        "Failed to initialize Wi-Fi driver"
    );


    // ---------------------------------
    // Event group
    // ---------------------------------

    wifi_event_group =
        xEventGroupCreate();

    if (wifi_event_group == NULL)
    {
        ESP_LOGE(
            TAG,
            "Failed to create Wi-Fi event group"
        );

        return ESP_ERR_NO_MEM;
    }


    // ---------------------------------
    // Register event handlers
    // ---------------------------------

    ESP_RETURN_ON_ERROR(
        esp_event_handler_register(
            WIFI_EVENT,
            ESP_EVENT_ANY_ID,
            &wifi_event_handler,
            NULL
        ),
        TAG,
        "Failed to register Wi-Fi event handler"
    );

    ESP_RETURN_ON_ERROR(
        esp_event_handler_register(
            IP_EVENT,
            IP_EVENT_STA_GOT_IP,
            &wifi_event_handler,
            NULL
        ),
        TAG,
        "Failed to register IP event handler"
    );


    // ---------------------------------
    // Wi-Fi credentials
    // ---------------------------------

    wifi_config_t wifi_config = {
        .sta = {
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        }
    };

    strncpy(
        (char *)wifi_config.sta.ssid,
        WIFI_SSID,
        sizeof(wifi_config.sta.ssid)
    );

    strncpy(
        (char *)wifi_config.sta.password,
        WIFI_PASSWORD,
        sizeof(wifi_config.sta.password)
    );


    // ---------------------------------
    // Configure Station mode
    // ---------------------------------

    ESP_RETURN_ON_ERROR(
        esp_wifi_set_mode(WIFI_MODE_STA),
        TAG,
        "Failed to set Wi-Fi mode"
    );

    ESP_RETURN_ON_ERROR(
        esp_wifi_set_config(
            WIFI_IF_STA,
            &wifi_config
        ),
        TAG,
        "Failed to configure Wi-Fi"
    );


    // ---------------------------------
    // Start Wi-Fi
    // ---------------------------------

    ESP_RETURN_ON_ERROR(
        esp_wifi_start(),
        TAG,
        "Failed to start Wi-Fi"
    );
    ESP_RETURN_ON_ERROR(
        esp_wifi_set_ps(WIFI_PS_NONE),
        TAG,
        "Failed to disable Wi-Fi power saving"
    );

    ESP_LOGI(TAG, "Wi-Fi power saving disabled");

    ESP_LOGI(
        TAG,
        "Connecting to SSID: %s",
        WIFI_SSID
    );


    // ---------------------------------
    // Wait until IP address is obtained
    // ---------------------------------

    EventBits_t bits =
        xEventGroupWaitBits(
            wifi_event_group,
            WIFI_CONNECTED_BIT,
            pdFALSE,
            pdTRUE,
            portMAX_DELAY
        );

    if (bits & WIFI_CONNECTED_BIT)
    {
        ESP_LOGI(
            TAG,
            "Network connection ready"
        );

        return ESP_OK;
    }


    return ESP_FAIL;
}


// ---------------------------------
// Check connection state
// ---------------------------------

bool wifi_manager_is_connected(void)
{
    return connected;
}