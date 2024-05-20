from .. import core

@core.extern
def globaltimer(_builder=None):
    return core.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;",
        "=l",
        [],
        dtype=core.int64,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@core.extern
def smid(_builder=None):
    return core.inline_asm_elementwise(
        "mov.u32 $0, %smid;",
        "=r",
        [],
        dtype=core.int32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@core.builtin
def num_threads(_builder=None):
    return core.constexpr(_builder.target.num_warps * 32)


@core.builtin
def num_warps(_builder=None):
    return core.constexpr(_builder.options.num_warps)


# This is a hack to put the fp8e4m3 to replace fp8e4b15 in triton
@core.builtin
def convert_uint8_as_fp8e4m3_to_float16(r0, _builder=None):
    return core.inline_asm_elementwise(
        "{                                      \n"
        ".reg .b32 a<2>, b<2>;                  \n"  # if input = 0xf1f2f3f4
        "prmt.b32 a0, 0, $2, 0x5040;            \n"  # a0 = 0xf300f400
        "prmt.b32 a1, 0, $2, 0x7060;            \n"  # a1 = 0xf100f200
        "lop3.b32 b0, a0, 0x7fff7fff, 0, 0xc0;  \n"  # b0 = a0 & 0x7fff7fff
        "lop3.b32 b1, a1, 0x7fff7fff, 0, 0xc0;  \n"  # (strip sign)
        "shr.b32  b0, b0, 1;                    \n"  # b0 >>= 1
        "shr.b32  b1, b1, 1;                    \n"  # shift into fp16 position
        "add.u32  b0, b0, 0x20002000;           \n"  # b0.exp += 2**4-2**3
        # exponent compensate = 8
        "add.u32  b1, b1, 0x20002000;           \n"  # b1 += 8<<10 | 8<<10<<16
        "lop3.b32 $0, b0, 0x80008000, a0, 0xf8; \n"  # out0 = b0|(0x80008000&a0)
        "lop3.b32 $1, b1, 0x80008000, a1, 0xf8; \n" # (restore sign)
        "}                                      \n",
        "=r,=r,r",
        [r0],
        dtype=core.float16,
        is_pure=True,
        pack=4,
        _builder=_builder,
    )
