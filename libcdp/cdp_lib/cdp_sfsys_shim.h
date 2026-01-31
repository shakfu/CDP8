/*
 * CDP SFSYS Shim - Intercepts file I/O for library mode
 *
 * When CDP_LIBRARY_MODE is defined, this header redirects sfsys I/O
 * functions to our memory-based implementations.
 */

#ifndef CDP_SFSYS_SHIM_H
#define CDP_SFSYS_SHIM_H

#ifdef CDP_LIBRARY_MODE

#include "cdp_io_redirect.h"

/*
 * Check if we should redirect this file descriptor.
 * Returns 1 if fd is a redirected memory buffer, 0 otherwise.
 */
static inline int cdp_should_redirect(int fd) {
    return g_cdp_io.library_mode && cdp_io_is_redirected(fd);
}

/*
 * Wrapper for fgetfbufEx - read float samples
 */
static inline int cdp_fgetfbufEx(float *fp, int count, int sfd, int expect_floats) {
    if (cdp_should_redirect(sfd)) {
        return cdp_io_read(sfd, fp, count);
    }
    /* Call original function */
    extern int fgetfbufEx(float *fp, int count, int sfd, int expect_floats);
    return fgetfbufEx(fp, count, sfd, expect_floats);
}

/*
 * Wrapper for fputfbufEx - write float samples
 */
static inline int cdp_fputfbufEx(float *fp, int count, int sfd) {
    if (cdp_should_redirect(sfd)) {
        return cdp_io_write(sfd, fp, count);
    }
    /* Call original function */
    extern int fputfbufEx(float *fp, int count, int sfd);
    return fputfbufEx(fp, count, sfd);
}

/*
 * Wrapper for sndseekEx
 */
static inline int cdp_sndseekEx(int sfd, int dist, int whence) {
    if (cdp_should_redirect(sfd)) {
        return cdp_io_seek(sfd, dist, whence);
    }
    extern int sndseekEx(int sfd, int dist, int whence);
    return sndseekEx(sfd, dist, whence);
}

/*
 * Wrapper for sndsizeEx
 */
static inline int cdp_sndsizeEx(int sfd) {
    if (cdp_should_redirect(sfd)) {
        return cdp_io_size(sfd);
    }
    extern int sndsizeEx(int sfd);
    return sndsizeEx(sfd);
}

/*
 * Wrapper for sndcloseEx
 */
static inline int cdp_sndcloseEx(int sfd) {
    if (cdp_should_redirect(sfd)) {
        cdp_io_close_slot(sfd);
        return 0;
    }
    extern int sndcloseEx(int sfd);
    return sndcloseEx(sfd);
}

/* Redefine sfsys functions to use our wrappers */
#define fgetfbufEx cdp_fgetfbufEx
#define fputfbufEx cdp_fputfbufEx
#define sndseekEx  cdp_sndseekEx
#define sndsizeEx  cdp_sndsizeEx
#define sndcloseEx cdp_sndcloseEx

#endif /* CDP_LIBRARY_MODE */

#endif /* CDP_SFSYS_SHIM_H */
