/**
  ******************************************************************************
  * @file    system_init.h
  * @brief   System initialization module for STM32U545RE-Q
  ******************************************************************************
  */

#ifndef __SYSTEM_INIT_H
#define __SYSTEM_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

/* System initialization functions */
void SystemClock_Config(void);
void SystemPower_Config(void);
void BSP_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* __SYSTEM_INIT_H */
