#ifndef DEBUG_LOG_H
#define DEBUG_LOG_H

// Debug logging functions using semihosting
// Output will appear in OpenOCD console

void debug_print(const char *str);
void debug_printf(const char *format, ...);

#endif // DEBUG_LOG_H
