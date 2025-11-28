#ifndef DEBUG_LOG_H
#define DEBUG_LOG_H

#ifdef DEBUG_SEMIHOSTING
    #include <stdio.h>
    // Map debug_printf to standard printf when semihosting is enabled
    #define debug_printf(...) printf(__VA_ARGS__)
    
    // Function prototype for semihosting init
    extern void initialise_monitor_handles(void);
#else
    // Optimize out debug_printf when debug is disabled
    #define debug_printf(...) do {} while(0)
#endif

#endif // DEBUG_LOG_H
