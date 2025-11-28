// Semihosting debug logging for STM32
// Enables printf() to work via the debug probe

#include <stdio.h>
#include <stdarg.h>

// Semihosting syscall numbers
#define SYS_WRITEC 0x03
#define SYS_WRITE0 0x04
#define SYS_WRITE  0x05

// ARM semihosting call
static inline int __attribute__((always_inline))
semihost_call(int reason, void *arg) {
    int result;
    __asm__ volatile (
        "mov r0, %[rsn]\n"
        "mov r1, %[arg]\n"
        "bkpt 0xAB\n"
        "mov %[res], r0\n"
        : [res] "=r" (result)
        : [rsn] "r" (reason), [arg] "r" (arg)
        : "r0", "r1", "memory"
    );
    return result;
}

// Write a null-terminated string via semihosting
void debug_print(const char *str) {
    semihost_call(SYS_WRITE0, (void*)str);
}

// Printf-style debug logging
void debug_printf(const char *format, ...) {
    char buffer[256];
    va_list args;
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    debug_print(buffer);
}

// Override _write for printf support
int _write(int file, char *ptr, int len) {
    (void)file;
    for (int i = 0; i < len; i++) {
        char c = ptr[i];
        semihost_call(SYS_WRITEC, &c);
    }
    return len;
}
