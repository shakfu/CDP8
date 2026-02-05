/*
 * CDP Constrict - Silence Constriction Effect
 *
 * Shortens the duration of zero-level (silent) sections in sound.
 */

#ifndef CDP_CONSTRICT_H
#define CDP_CONSTRICT_H

#include "cdp_lib.h"

/*
 * Apply constrict effect - shorten or remove silent sections.
 *
 * Scans through audio looking for zero-value samples and reduces
 * or removes these silent sections based on the constriction parameter.
 *
 * Parameters:
 *   ctx          - Processing context
 *   input        - Input buffer
 *   constriction - Percentage of silence removal (0.0 to 200.0):
 *                  0-100: Shorten zero-sections by that percentage
 *                         (e.g., 50 = silences are 50% shorter)
 *                  100-200: Overlap sounds on either side of silence
 *                           by (constriction - 100)%
 *                           (e.g., 150 = 50% overlap, causing sounds to merge)
 *
 * Returns:
 *   New buffer with constricted silences, or NULL on error.
 */
cdp_lib_buffer* cdp_lib_constrict(cdp_lib_ctx* ctx,
                                   const cdp_lib_buffer* input,
                                   double constriction);

#endif /* CDP_CONSTRICT_H */
