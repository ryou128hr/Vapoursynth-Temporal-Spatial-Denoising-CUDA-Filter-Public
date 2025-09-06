#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include "denoiser_cuda.h"

typedef struct DenoiseData {
    VSNode* node;
    VSVideoInfo vi;
    int strength;   // 空間デノイズ強度 (0-100)
    float temporal; // 時間方向の重み (0.0-1.0)
} DenoiseData;


static const VSFrame* VS_CC simpleDenoiseGetFrame(
    int n, int activationReason, void* instanceData, void**,
    VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi)
{
    DenoiseData* d = (DenoiseData*)instanceData;
    const int max_f = d->vi.numFrames - 1;

    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        if (n > 0) vsapi->requestFrameFilter(n - 1, d->node, frameCtx);
        if (n < max_f) vsapi->requestFrameFilter(n + 1, d->node, frameCtx);
        return nullptr;
    }

    if (activationReason == arAllFramesReady) {
        const VSFrame* cur_f = vsapi->getFrameFilter(n, d->node, frameCtx);
        const VSFrame* prev_f = (n > 0) ? vsapi->getFrameFilter(n - 1, d->node, frameCtx) : nullptr;
        const VSFrame* next_f = (n < max_f) ? vsapi->getFrameFilter(n + 1, d->node, frameCtx) : nullptr;

        VSFrame* dst = vsapi->newVideoFrame(&d->vi.format,
            d->vi.width, d->vi.height,
            cur_f, core);

        for (int plane = 0; plane < d->vi.format.numPlanes; ++plane) {
            const int h = vsapi->getFrameHeight(cur_f, plane);
            const int w = vsapi->getFrameWidth(cur_f, plane);
            const ptrdiff_t stride = vsapi->getStride(cur_f, plane);
            const uint8_t* curp = vsapi->getReadPtr(cur_f, plane);
            const uint8_t* prevp = prev_f ? vsapi->getReadPtr(prev_f, plane) : nullptr;
            const uint8_t* nextp = next_f ? vsapi->getReadPtr(next_f, plane) : nullptr;
            uint8_t* dstp = vsapi->getWritePtr(dst, plane);

            run_simple_denoise(curp, prevp, nextp, dstp,
                w, h, (int)stride,
                d->strength, d->temporal);
        }

        if (prev_f) vsapi->freeFrame(prev_f);
        if (next_f) vsapi->freeFrame(next_f);
        vsapi->freeFrame(cur_f);

        return dst;
    }

    return nullptr;
}

static void VS_CC simpleDenoiseFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    DenoiseData* d = (DenoiseData*)instanceData;
    if (d->node) vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC simpleDenoiseCreate(const VSMap* in, VSMap* out, void* userData,
    VSCore* core, const VSAPI* vsapi) {
    int err = 0;
    DenoiseData* d = (DenoiseData*)malloc(sizeof(DenoiseData));
    d->node = vsapi->mapGetNode(in, "clip", 0, &err);
    if (err) { vsapi->mapSetError(out, "SimpleDenoiseCUDA: clip is required."); free(d); return; }

    d->vi = *vsapi->getVideoInfo(d->node);

    d->strength = (int)vsapi->mapGetInt(in, "strength", 0, &err);
    if (err) d->strength = 20; // default
    d->strength = std::clamp(d->strength, 0, 100);

    d->temporal = (float)vsapi->mapGetFloat(in, "temporal", 0, &err);
    if (err) d->temporal = 0.4f; // default
    d->temporal = std::clamp(d->temporal, 0.0f, 1.0f);

    VSFilterDependency deps[] = { { d->node, rpGeneral } };
    vsapi->createVideoFilter(out, "SimpleDenoiseCUDA", &d->vi,
        simpleDenoiseGetFrame, simpleDenoiseFree,
        fmParallel, deps, 1, d, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.example.simpledenoisecuda", "sden",
        "Simple Spatial+Temporal Denoiser (CUDA)",
        VS_MAKE_VERSION(1, 0),
        VAPOURSYNTH_API_VERSION,
        0, plugin);
    vspapi->registerFunction("SimpleDenoiseCUDA",
        "clip:vnode;strength:int:opt;temporal:float:opt;",
        "clip:vnode;",
        simpleDenoiseCreate, NULL, plugin);
}
