/*
 * CDP Shim Layer - Memory buffer I/O for CDP processing functions.
 *
 * This replaces file-based I/O with memory buffer operations,
 * allowing CDP algorithms to be used as library functions.
 */

#ifndef CDP_SHIM_H
#define CDP_SHIM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration - actual structure defined in structures.h */
typedef struct datalist *dataptr;

/*
 * Memory buffer descriptor for I/O redirection.
 */
typedef struct cdp_membuf {
    float *data;           /* Sample data */
    size_t capacity;       /* Total capacity in samples */
    size_t length;         /* Actual data length in samples */
    size_t position;       /* Current read/write position */
    int channels;
    int sample_rate;
} cdp_membuf;

/*
 * CDP shim context - holds state for memory-based I/O.
 */
typedef struct cdp_shim_ctx {
    cdp_membuf *input;     /* Input buffer(s) */
    int input_count;       /* Number of input buffers */
    cdp_membuf *output;    /* Output buffer */
    int initialized;
} cdp_shim_ctx;

/* Global shim context (CDP uses global state) */
extern cdp_shim_ctx *g_cdp_shim;

/*
 * Initialize the shim layer.
 * Must be called before any CDP processing.
 */
int cdp_shim_init(void);

/*
 * Clean up the shim layer.
 */
void cdp_shim_cleanup(void);

/*
 * Set input buffer for processing.
 * The shim will redirect file reads to this buffer.
 */
int cdp_shim_set_input(float *data, size_t length, int channels, int sample_rate);

/*
 * Set output buffer for processing.
 * The shim will redirect file writes to this buffer.
 * If capacity is 0, the shim will allocate as needed.
 */
int cdp_shim_set_output(float *data, size_t capacity, int channels, int sample_rate);

/*
 * Get output buffer after processing.
 * Returns the number of samples written.
 */
size_t cdp_shim_get_output(float **data, int *channels, int *sample_rate);

/*
 * Shim I/O functions - these replace sfsys functions.
 */

/* Replacement for sndopenEx - returns fake file descriptor */
int shim_sndopenEx(const char *name, int auto_scale, int access);

/* Replacement for sndcreat_formatted */
int shim_sndcreat_formatted(const char *fn, int size, int stype,
                            int channels, int srate, int mode);

/* Replacement for sndcloseEx */
int shim_sndcloseEx(int sfd);

/* Replacement for fgetfbufEx - read samples from input buffer */
int shim_fgetfbufEx(float *fp, int count, int sfd, int expect_floats);

/* Replacement for fputfbufEx - write samples to output buffer */
int shim_fputfbufEx(float *fp, int count, int sfd);

/* Replacement for sndseekEx */
int shim_sndseekEx(int sfd, int dist, int whence);

/* Replacement for sndsizeEx */
int shim_sndsizeEx(int sfd);

/* File descriptor constants for shim */
#define SHIM_INPUT_FD  1000
#define SHIM_OUTPUT_FD 1001

#ifdef __cplusplus
}
#endif

#endif /* CDP_SHIM_H */
