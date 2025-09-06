# Vapoursynth-Temporal-Spatial-Denoising-CUDA-Filter-Public
CUDAで処理する空間／時間　デノイザー。

運用環境：CUDA 12.6  RTX 3080TI

必須：CUDA 12.6 

cudart64_12.dllも同じフォルダに入れてください。


```

denoised = core.sden.SimpleDenoiseCUDA(
    clip, 
    strength=55,   # 空間デノイズ強度
    temporal=27   # 時間方向の重み
)

```
strength=空間デノイズ。多いほどぼやけますがノイズが取れます。
temporal＝多いほど時間軸デノイズされます。
