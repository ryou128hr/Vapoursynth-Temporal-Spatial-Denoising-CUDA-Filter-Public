#pragma once
#include <cstdint>
#include <stddef.h>

// CUDA �������ɂ���֐���錾
void run_simple_denoise(
    const uint8_t* cur,
    const uint8_t* prev,
    const uint8_t* next,
    uint8_t* dst,
    int w, int h, int stride,
    int strength, float temporal);
