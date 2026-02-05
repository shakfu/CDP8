/*
 * CDP I/O Redirect Layer - Implementation
 */

#include "cdp_io_redirect.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global I/O state */
cdp_io_state g_cdp_io = {0};

/* Base file descriptor for redirected slots */
#define CDP_FD_BASE 10000

int cdp_io_init(void) {
    if (g_cdp_io.initialized) {
        return 0;
    }

    memset(&g_cdp_io, 0, sizeof(cdp_io_state));
    g_cdp_io.next_fd = CDP_FD_BASE;
    g_cdp_io.initialized = 1;
    g_cdp_io.library_mode = 0;  /* Disabled by default */

    return 0;
}

void cdp_io_cleanup(void) {
    if (!g_cdp_io.initialized) {
        return;
    }

    /* Free any allocated buffers */
    for (int i = 0; i < CDP_MAX_IO_SLOTS; i++) {
        cdp_io_slot *slot = &g_cdp_io.slots[i];
        if (slot->in_use && slot->owns_data && slot->data != NULL) {
            free(slot->data);
        }
        memset(slot, 0, sizeof(cdp_io_slot));
    }

    g_cdp_io.initialized = 0;
}

void cdp_io_set_library_mode(int enabled) {
    if (!g_cdp_io.initialized) {
        cdp_io_init();
    }
    g_cdp_io.library_mode = enabled ? 1 : 0;
}

static cdp_io_slot* find_free_slot(void) {
    for (int i = 0; i < CDP_MAX_IO_SLOTS; i++) {
        if (!g_cdp_io.slots[i].in_use) {
            return &g_cdp_io.slots[i];
        }
    }
    return NULL;
}

int cdp_io_register_input(float *data, size_t length, int channels, int sample_rate) {
    if (!g_cdp_io.initialized) {
        cdp_io_init();
    }

    cdp_io_slot *slot = find_free_slot();
    if (slot == NULL) {
        snprintf(g_cdp_io.error_msg, sizeof(g_cdp_io.error_msg),
                 "No free I/O slots available");
        return -1;
    }

    slot->in_use = 1;
    slot->is_input = 1;
    slot->data = data;
    slot->capacity = length;
    slot->length = length;
    slot->position = 0;
    slot->channels = channels;
    slot->sample_rate = sample_rate;
    slot->sample_type = 1;  /* SAMP_FLOAT */
    slot->owns_data = 0;    /* Caller owns the data */

    return g_cdp_io.next_fd++;
}

int cdp_io_register_output(float *data, size_t capacity, int channels, int sample_rate) {
    if (!g_cdp_io.initialized) {
        cdp_io_init();
    }

    cdp_io_slot *slot = find_free_slot();
    if (slot == NULL) {
        snprintf(g_cdp_io.error_msg, sizeof(g_cdp_io.error_msg),
                 "No free I/O slots available");
        return -1;
    }

    slot->in_use = 1;
    slot->is_input = 0;
    slot->channels = channels;
    slot->sample_rate = sample_rate;
    slot->sample_type = 1;  /* SAMP_FLOAT */
    slot->position = 0;
    slot->length = 0;

    if (data != NULL) {
        slot->data = data;
        slot->capacity = capacity;
        slot->owns_data = 0;
    } else {
        /* Allocate initial buffer */
        size_t initial_capacity = capacity > 0 ? capacity : 65536;
        slot->data = (float *)malloc(initial_capacity * sizeof(float));
        if (slot->data == NULL) {
            slot->in_use = 0;
            return -1;
        }
        slot->capacity = initial_capacity;
        slot->owns_data = 1;
    }

    return g_cdp_io.next_fd++;
}

size_t cdp_io_get_output(int fd, float **data, int *channels, int *sample_rate) {
    cdp_io_slot *slot = cdp_io_get_slot(fd);
    if (slot == NULL || slot->is_input) {
        return 0;
    }

    *data = slot->data;
    *channels = slot->channels;
    *sample_rate = slot->sample_rate;

    return slot->length;
}

void cdp_io_close_slot(int fd) {
    cdp_io_slot *slot = cdp_io_get_slot(fd);
    if (slot == NULL) {
        return;
    }

    if (slot->owns_data && slot->data != NULL) {
        free(slot->data);
    }

    memset(slot, 0, sizeof(cdp_io_slot));
}

int cdp_io_is_redirected(int fd) {
    return fd >= CDP_FD_BASE && fd < g_cdp_io.next_fd;
}

cdp_io_slot* cdp_io_get_slot(int fd) {
    if (!cdp_io_is_redirected(fd)) {
        return NULL;
    }

    int index = fd - CDP_FD_BASE;
    if (index < 0 || index >= CDP_MAX_IO_SLOTS) {
        return NULL;
    }

    cdp_io_slot *slot = &g_cdp_io.slots[index];
    if (!slot->in_use) {
        return NULL;
    }

    return slot;
}

int cdp_io_read(int fd, float *buf, int count) {
    cdp_io_slot *slot = cdp_io_get_slot(fd);
    if (slot == NULL) {
        return -1;
    }

    size_t available = slot->length - slot->position;
    size_t to_read = (size_t)count;
    if (to_read > available) {
        to_read = available;
    }

    if (to_read == 0) {
        return 0;  /* EOF */
    }

    memcpy(buf, slot->data + slot->position, to_read * sizeof(float));
    slot->position += to_read;

    return (int)to_read;
}

int cdp_io_write(int fd, float *buf, int count) {
    cdp_io_slot *slot = cdp_io_get_slot(fd);
    if (slot == NULL) {
        return -1;
    }

    /* Check if we need to grow the buffer */
    size_t needed = slot->position + (size_t)count;
    if (needed > slot->capacity) {
        if (!slot->owns_data) {
            /* Can't grow a buffer we don't own */
            snprintf(g_cdp_io.error_msg, sizeof(g_cdp_io.error_msg),
                     "Output buffer overflow");
            return -1;
        }

        /* Grow the buffer */
        size_t new_capacity = slot->capacity * 2;
        while (new_capacity < needed) {
            new_capacity *= 2;
        }

        float *new_data = (float *)realloc(slot->data, new_capacity * sizeof(float));
        if (new_data == NULL) {
            return -1;
        }

        slot->data = new_data;
        slot->capacity = new_capacity;
    }

    memcpy(slot->data + slot->position, buf, (size_t)count * sizeof(float));
    slot->position += (size_t)count;

    if (slot->position > slot->length) {
        slot->length = slot->position;
    }

    return count;
}

int cdp_io_seek(int fd, int offset, int whence) {
    cdp_io_slot *slot = cdp_io_get_slot(fd);
    if (slot == NULL) {
        return -1;
    }

    size_t new_pos;
    switch (whence) {
        case 0:  /* SEEK_SET */
            new_pos = (size_t)offset;
            break;
        case 1:  /* SEEK_CUR */
            new_pos = slot->position + (size_t)offset;
            break;
        case 2:  /* SEEK_END */
            new_pos = slot->length + (size_t)offset;
            break;
        default:
            return -1;
    }

    /* Clamp to valid range */
    if (new_pos > slot->length) {
        new_pos = slot->length;
    }

    slot->position = new_pos;
    return (int)new_pos;
}

int cdp_io_tell(int fd) {
    cdp_io_slot *slot = cdp_io_get_slot(fd);
    if (slot == NULL) {
        return -1;
    }
    return (int)slot->position;
}

int cdp_io_size(int fd) {
    cdp_io_slot *slot = cdp_io_get_slot(fd);
    if (slot == NULL) {
        return -1;
    }
    return (int)slot->length;
}
