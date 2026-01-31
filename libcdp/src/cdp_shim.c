/*
 * CDP Shim Layer - Memory buffer I/O implementation.
 */

#include "cdp_shim.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Global shim context */
cdp_shim_ctx *g_cdp_shim = NULL;

/* CDP global error string - used by mxfft.c */
char errstr[2400];

/* Internal input/output buffers */
static cdp_membuf g_input_buf;
static cdp_membuf g_output_buf;

int cdp_shim_init(void) {
    if (g_cdp_shim != NULL) {
        return 0;  /* Already initialized */
    }

    g_cdp_shim = (cdp_shim_ctx *)calloc(1, sizeof(cdp_shim_ctx));
    if (g_cdp_shim == NULL) {
        return -1;
    }

    memset(&g_input_buf, 0, sizeof(cdp_membuf));
    memset(&g_output_buf, 0, sizeof(cdp_membuf));

    g_cdp_shim->input = &g_input_buf;
    g_cdp_shim->input_count = 1;
    g_cdp_shim->output = &g_output_buf;
    g_cdp_shim->initialized = 1;

    return 0;
}

void cdp_shim_cleanup(void) {
    if (g_cdp_shim == NULL) {
        return;
    }

    /* Free output buffer if we allocated it */
    if (g_output_buf.data != NULL && g_output_buf.capacity > 0) {
        /* Only free if we own it (capacity was set by shim) */
        /* For now, don't free - caller manages memory */
    }

    memset(&g_input_buf, 0, sizeof(cdp_membuf));
    memset(&g_output_buf, 0, sizeof(cdp_membuf));

    free(g_cdp_shim);
    g_cdp_shim = NULL;
}

int cdp_shim_set_input(float *data, size_t length, int channels, int sample_rate) {
    if (g_cdp_shim == NULL) {
        if (cdp_shim_init() != 0) {
            return -1;
        }
    }

    g_input_buf.data = data;
    g_input_buf.capacity = length;
    g_input_buf.length = length;
    g_input_buf.position = 0;
    g_input_buf.channels = channels;
    g_input_buf.sample_rate = sample_rate;

    return 0;
}

int cdp_shim_set_output(float *data, size_t capacity, int channels, int sample_rate) {
    if (g_cdp_shim == NULL) {
        if (cdp_shim_init() != 0) {
            return -1;
        }
    }

    g_output_buf.data = data;
    g_output_buf.capacity = capacity;
    g_output_buf.length = 0;
    g_output_buf.position = 0;
    g_output_buf.channels = channels;
    g_output_buf.sample_rate = sample_rate;

    return 0;
}

size_t cdp_shim_get_output(float **data, int *channels, int *sample_rate) {
    if (g_cdp_shim == NULL || g_output_buf.data == NULL) {
        return 0;
    }

    *data = g_output_buf.data;
    *channels = g_output_buf.channels;
    *sample_rate = g_output_buf.sample_rate;

    return g_output_buf.length;
}

/*
 * Shim I/O Functions
 */

int shim_sndopenEx(const char *name, int auto_scale, int access) {
    (void)name;
    (void)auto_scale;
    (void)access;

    /* Return fake file descriptor for input */
    return SHIM_INPUT_FD;
}

int shim_sndcreat_formatted(const char *fn, int size, int stype,
                            int channels, int srate, int mode) {
    (void)fn;
    (void)size;
    (void)stype;
    (void)mode;

    /* Configure output buffer properties */
    g_output_buf.channels = channels;
    g_output_buf.sample_rate = srate;

    /* Return fake file descriptor for output */
    return SHIM_OUTPUT_FD;
}

int shim_sndcloseEx(int sfd) {
    (void)sfd;
    /* Nothing to do for memory buffers */
    return 0;
}

int shim_fgetfbufEx(float *fp, int count, int sfd, int expect_floats) {
    (void)expect_floats;

    if (sfd != SHIM_INPUT_FD || g_cdp_shim == NULL) {
        return -1;
    }

    cdp_membuf *buf = g_cdp_shim->input;
    if (buf->data == NULL) {
        return -1;
    }

    /* Calculate how many samples we can read */
    size_t available = buf->length - buf->position;
    size_t to_read = (size_t)count;
    if (to_read > available) {
        to_read = available;
    }

    if (to_read == 0) {
        return 0;  /* EOF */
    }

    /* Copy data */
    memcpy(fp, buf->data + buf->position, to_read * sizeof(float));
    buf->position += to_read;

    return (int)to_read;
}

int shim_fputfbufEx(float *fp, int count, int sfd) {
    if (sfd != SHIM_OUTPUT_FD || g_cdp_shim == NULL) {
        return -1;
    }

    cdp_membuf *buf = g_cdp_shim->output;

    /* Check if we need to grow the buffer */
    size_t needed = buf->position + (size_t)count;
    if (needed > buf->capacity) {
        /* Need to reallocate */
        size_t new_capacity = buf->capacity == 0 ? 65536 : buf->capacity * 2;
        while (new_capacity < needed) {
            new_capacity *= 2;
        }

        float *new_data = (float *)realloc(buf->data, new_capacity * sizeof(float));
        if (new_data == NULL) {
            return -1;
        }

        buf->data = new_data;
        buf->capacity = new_capacity;
    }

    /* Copy data */
    memcpy(buf->data + buf->position, fp, (size_t)count * sizeof(float));
    buf->position += (size_t)count;

    if (buf->position > buf->length) {
        buf->length = buf->position;
    }

    return count;
}

int shim_sndseekEx(int sfd, int dist, int whence) {
    cdp_membuf *buf = NULL;

    if (sfd == SHIM_INPUT_FD) {
        buf = g_cdp_shim ? g_cdp_shim->input : NULL;
    } else if (sfd == SHIM_OUTPUT_FD) {
        buf = g_cdp_shim ? g_cdp_shim->output : NULL;
    }

    if (buf == NULL) {
        return -1;
    }

    size_t new_pos;
    switch (whence) {
        case 0:  /* SEEK_SET */
            new_pos = (size_t)dist;
            break;
        case 1:  /* SEEK_CUR */
            new_pos = buf->position + (size_t)dist;
            break;
        case 2:  /* SEEK_END */
            new_pos = buf->length + (size_t)dist;
            break;
        default:
            return -1;
    }

    if (new_pos > buf->length) {
        new_pos = buf->length;
    }

    buf->position = new_pos;
    return (int)new_pos;
}

int shim_sndsizeEx(int sfd) {
    cdp_membuf *buf = NULL;

    if (sfd == SHIM_INPUT_FD) {
        buf = g_cdp_shim ? g_cdp_shim->input : NULL;
    } else if (sfd == SHIM_OUTPUT_FD) {
        buf = g_cdp_shim ? g_cdp_shim->output : NULL;
    }

    if (buf == NULL) {
        return -1;
    }

    return (int)buf->length;
}
