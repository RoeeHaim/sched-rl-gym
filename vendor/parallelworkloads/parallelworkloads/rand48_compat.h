#ifndef RAND48_COMPAT_H
#define RAND48_COMPAT_H

/* MSVC lacks the POSIX rand48 family and bzero. The implementation below
 * follows the POSIX drand48 specification (48-bit LCG, a=0x5DEECE66D, c=0xB)
 * so generated workloads are bit-identical to those produced on Linux. */
#if defined(_WIN32) || defined(_MSC_VER)

#include <stdint.h>
#include <string.h>

#define bzero(p, n) memset((p), 0, (n))

static uint64_t _rand48_state = 0x330E;

static void srand48(long seed)
{
    _rand48_state = (((uint64_t)(uint32_t)seed) << 16) | 0x330E;
}

static uint64_t _rand48_next(void)
{
    _rand48_state =
        (0x5DEECE66DULL * _rand48_state + 0xBULL) & ((1ULL << 48) - 1);
    return _rand48_state;
}

static double drand48(void)
{
    return (double)_rand48_next() / (double)(1ULL << 48);
}

static long lrand48(void)
{
    return (long)(_rand48_next() >> 17);
}

#endif /* _WIN32 || _MSC_VER */

#endif /* RAND48_COMPAT_H */
