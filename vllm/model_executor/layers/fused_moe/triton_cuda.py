from .. import core


@core.extern
def globaltimer(_builder=None):
    return core.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=core.int64, is_pure=False, pack=1,
                                       _builder=_builder)


@core.extern
def smid(_builder=None):
    return core.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=core.int32, is_pure=True, pack=1,
                                       _builder=_builder)


@core.builtin
def num_threads(_builder=None):
    return core.constexpr(_builder.target.num_warps * 32)

@core.builtin
def num_warps(_builder=None):
    return core.constexpr(_builder.options.num_warps)


# ----- FP8E4M3B15 ------
# This is a hack to put the fp8e4m3 to replace fp8e4b15 in triton
@core.builtin
def convert_fp8e4b15_to_float16(arg, _builder=None):
    return core.extern_elementwise(
        "libdevice", libdevice_path(), [arg0, 0], {
            (core.dtype("fp16"), ): ("__nv_cvt_fp8_to_halfraw", core.dtype("fp8e4b15"), core.dtype("int32")),
        }, is_pure=True, _builder=_builder)


@core.builtin
def convert_float16_to_fp8e4b15(arg, has_minx2, _builder=None):
    raise NotImplementedError("convert_float16_to_fp8e4b15 is not implemented")


@core.builtin
def convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2, _builder=None):
    if arg.type.scalar.is_fp8e4b15():
        upcast_val = convert_fp8e4b15_to_float16(arg, _builder=_builder)
        if dst_ty.scalar.is_fp32():
            upcast_val = upcast_val.to(core.float32, _builder=_builder)
        return upcast_val

    assert arg.type.scalar.is_fp16() or arg.type.scalar.is_fp32()
    downcast_val = arg
    if arg.type.scalar.is_fp32():
        downcast_val = downcast_val.to(core.float16, fp_downcast_rounding="rtz", _builder=_builder)
    downcast_val = convert_float16_to_fp8e4b15(downcast_val, has_minx2=has_minx2, _builder=_builder)
    return downcast_val


@core.builtin
def convert_custom_float8_sm80(arg, dst_ty, fp_downcast_rounding=None, _builder=None):
    return convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2=True, _builder=_builder)


@core.builtin
def convert_custom_float8_sm70(arg, dst_ty, fp_downcast_rounding=None, _builder=None):
    return convert_custom_float8(arg, dst_ty, fp_downcast_rounding, has_minx2=False, _builder=_builder)